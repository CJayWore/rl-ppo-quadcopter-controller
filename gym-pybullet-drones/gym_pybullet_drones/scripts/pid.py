'''
Acknowledgements:
This script is adapted from an example in the gym-pybullet-drones package.
https://github.com/utiasDSL/gym-pybullet-drones/blob/main/gym_pybullet_drones/examples/pid.py

@INPROCEEDINGS{panerati2021learning,
      title={Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control}, 
      author={Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig},
      booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2021},
      volume={},
      number={},
      pages={7512-7519},
      doi={10.1109/IROS51168.2021.9635857}
}
'''

"""Script demonstrating PID control for hovering at target positions.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Usage Command:
    python pid.py --max_episodes 1000 --episode_timeout_sec 8 --target_hover_time 5.0 --gui False --randomize_positions True --use_best_params True --plot False --enable_performance_eval True
    --enable_noise True --noise_level heavy
    python pid.py --max_episodes 10 --episode_timeout_sec 8 --target_hover_time 5.0 --gui False --randomize_positions True --use_best_params True --plot False --enable_performance_eval True

Features:
    - Multi-drone simulation with independent PID controllers
    - Hovering at randomized target positions
    - Visualization options via PyBullet GUI
    - Data logging and plotting capabilities
    - Customizable simulation parameters

Notes:
    The drones start at randomized initial positions and navigate to randomized
    target positions, then hover there for the remainder of the simulation.

Outputs:
    - Simulation logs saved to the specified output folder
    - Optional CSV export of flight data
    - Optional plots showing drone positions, orientations, and control inputs
"""

import json
import os
import time
import argparse
import numpy as np
import pybullet as p
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

# Import performance evaluation from current directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'rl_framework'))
from performance_evaluation import DronePerformanceLogger
from gaussian_noise import GaussianNoiseManager, create_light_noise_manager, create_heavy_noise_manager, NoiseType


DEFAULT_DRONES = DroneModel("cf2p")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 30
DEFAULT_OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'pid_results')
DEFAULT_COLAB = False
DEFAULT_TRAJECTORY = "hover"  # Only hover mode supported

# Episode management constants
DEFAULT_MAX_EPISODES = 10
DEFAULT_EPISODE_TIMEOUT_SEC = 60
DEFAULT_TARGET_HOVER_TIME = 5.0  # Renamed for clarity
DEFAULT_POSITION_THRESHOLD = 0.1  # Added missing constant
DEFAULT_CRASH_ALTITUDE = 0.1  # Added missing constant
DEFAULT_MIN_ALTITUDE = 0.1
DEFAULT_MAX_ALTITUDE = 3.0

def check_episode_termination(obs, current_time, hover_duration, episode_timeout, 
                            required_hover_time, crash_altitude):
    """
    Check if episode should be terminated and return reason
    
    Returns:
        (terminated, reason): bool and string explaining termination
    """
    if len(obs) == 0:
        return True, "no_observation"
    
    # Extract state information from first drone
    state = obs[0]  # Shape should be (20,) for single drone observation
    pos = state[0:3]      # Position (x, y, z)
    quat = state[3:7]     # Quaternion (x, y, z, w)
    
    # Check crash (altitude too low)
    if pos[2] < crash_altitude:
        return True, "crash"
    
    # Check success (hovering long enough)
    if hover_duration >= required_hover_time:
        return True, "success"
    
    # Check timeout
    if current_time >= episode_timeout:
        return True, "timeout"
    
    return False, "running"

def update_hover_timer(obs, target_positions, hover_start_time, current_time, position_threshold):
    """
    Update hover timer based on whether drone is close to target
    
    Args:
        obs: Observation array from environment (shape: num_drones x obs_dim)
        target_positions: Array of target positions (shape: num_drones x 3)
        hover_start_time: When hovering started (or None)
        current_time: Current simulation time
        position_threshold: Distance threshold for considering drone "at target"
    
    Returns:
        (hover_duration, is_hovering): How long drone has been hovering and whether currently hovering
    """
    # Extract position from observation (first 3 elements for each drone)
    if len(obs) > 0:
        # obs is array of observations for each drone
        pos = obs[0][0:3]  # Position of first drone (x, y, z)
        target_pos = target_positions[0]  # Target for first drone
        
        distance_to_target = np.linalg.norm(pos - target_pos)
        
        if distance_to_target <= position_threshold:
            # Drone is at target
            if hover_start_time is not None:
                # Already hovering, return duration
                return current_time - hover_start_time, True
            else:
                # Just started hovering, return minimal time to indicate start
                return 0.01, True  # Small non-zero value to indicate hovering started
        else:
            # Drone not at target - not hovering
            return 0.0, False
    
    return 0.0, False

def load_best_pid_params(json_file="Best_PID_Params/best_pid_params.json"):
    """
    加载最佳PID参数
    优先加载PIDTuner生成的best_pid_params.json，如果不存在则加载默认参数
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir)
        
        json_path = os.path.join(project_root, json_file)
        json_path = os.path.normpath(json_path)  # 规范化路径
        
        print(f"🔍 Looking for PID parameters at: {json_path}")
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查数据格式 - 支持两种格式
            if 'best_params' in data:
                # 旧格式：参数在best_params字段中
                best_params = data['best_params']
                score = data.get('best_score', 'Unknown')
            else:
                # 新格式：直接包含PID参数
                best_params = {
                    'name': data.get('name', 'LoadedParams'),
                    'pos_p': data['pos_p'],
                    'pos_i': data['pos_i'],
                    'pos_d': data['pos_d'],
                    'att_p': data['att_p'],
                    'att_i': data['att_i'],
                    'att_d': data['att_d']
                }
                score = data.get('optimization_score', 'Unknown')
            
            print(f"🏆 Loading PID parameters (Score: {score})")
            print(f"📅 Generated: {data.get('timestamp', 'Unknown')}")
            print(f"Target: Parameter set: {best_params['name']}")
            print(f"🎪 Task type: {data.get('task_type', 'Unknown')}")
            
            return best_params
        else:
            print(f"Error: PID parameters file not found: {json_path}")
            print(f"💡 Trying fallback files...")
            
            # 尝试加载其他文件
            fallback_files = [
                "Best_PID_Params/recommended_hover.json",
                "Best_PID_Params/hover_conservative_pid.json"
            ]
            
            for fallback_file in fallback_files:
                fallback_path = os.path.join(project_root, fallback_file)
                if os.path.exists(fallback_path):
                    print(f"📄 Found fallback file: {fallback_path}")
                    return load_best_pid_params(fallback_file)
            
            print(f"💡 No parameter files found, using hardcoded defaults")
            # 返回保守的默认参数
            return {
                'name': 'HardcodedDefault',
                'pos_p': [0.4, 0.4, 0.6],
                'pos_i': [0.05, 0.05, 0.1],
                'pos_d': [0.2, 0.2, 0.3],
                'att_p': [8000, 8000, 6000],
                'att_i': [1.0, 1.0, 5.0],
                'att_d': [800, 800, 600]
            }
    except Exception as e:
        print(f"Error: Failed to load PID parameters: {e}")
        print(f"💡 Using default conservative parameters")
        return {
            'name': 'DefaultConservative',
            'pos_p': [0.4, 0.4, 0.6],
            'pos_i': [0.05, 0.05, 0.1],
            'pos_d': [0.2, 0.2, 0.3],
            'att_p': [8000, 8000, 6000],
            'att_i': [1.0, 1.0, 5.0],
            'att_d': [800, 800, 600]
        }

def visualize_targets(target_positions, pyb_client, target_radius=0.15):
    """
    可视化目标位置
    
    Parameters:
    - target_positions: numpy array of target positions [(x, y, z), ...]
    - pyb_client: PyBullet client ID
    - target_radius: radius of target area visualization
    
    Returns:
    - List of target visual object IDs
    """
    target_visual_ids = []
    
    for i, target_pos in enumerate(target_positions):
        # 创建目标点球体（绿色半透明）
        target_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.1,
            rgbaColor=[0, 1, 0, 0.8],  # 绿色半透明
            physicsClientId=pyb_client
        )
        
        target_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_visual,
            basePosition=target_pos,
            physicsClientId=pyb_client
        )
        
        # 创建目标区域（绿色圆环）
        target_area_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=target_radius,
            length=0.02,
            rgbaColor=[0, 1, 0, 0.3],  # 绿色透明
            physicsClientId=pyb_client
        )
        
        target_area_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_area_visual,
            basePosition=target_pos,
            physicsClientId=pyb_client
        )
        
        target_visual_ids.extend([target_id, target_area_id])
        
        # 添加目标标签
        p.addUserDebugText(
            text=f"Target {i}",
            textPosition=[target_pos[0], target_pos[1], target_pos[2] + 0.3],
            textColorRGB=[0, 1, 0],
            textSize=1.5,
            physicsClientId=pyb_client
        )
    
    return target_visual_ids

def draw_connection_lines(start_positions, target_positions, pyb_client):
    """
    绘制从起始位置到目标位置的连接线
    
    Parameters:
    - start_positions: numpy array of start positions
    - target_positions: numpy array of target positions
    - pyb_client: PyBullet client ID
    
    Returns:
    - List of line IDs
    """
    line_ids = []
    
    for i, (start_pos, target_pos) in enumerate(zip(start_positions, target_positions)):
        line_id = p.addUserDebugLine(
            lineFromXYZ=start_pos,
            lineToXYZ=target_pos,
            lineColorRGB=[0, 0, 1],  # 蓝色线条
            lineWidth=2,
            physicsClientId=pyb_client
        )
        line_ids.append(line_id)
    
    return line_ids
    
def generate_episode_positions(num_drones, randomize_positions):
    """Generate new start and target positions for each episode"""
    if randomize_positions:
        # Ultra-conservative random positions for maximum stability
        START_POSITIONS = []
        TARGET_POSITIONS = []
        
        for i in range(num_drones):
            # Very close start position for gentle startup
            start_pos = np.array([
                np.random.uniform(-0.5, 0.5),  # Even smaller range
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(0.8, 1.2)    # Higher start altitude
            ])
            
            # Closer target position for stable PID behavior
            while True:
                target_pos = np.array([
                    np.random.uniform(2.0, 4.0),
                    np.random.uniform(2.0, 4.0),
                    np.random.uniform(0.5, 2.0)
                ])
                # Minimum distance reduced for stability
                if np.linalg.norm(target_pos - start_pos) > 0.5:
                    break
            
            START_POSITIONS.append(start_pos)
            TARGET_POSITIONS.append(target_pos)
            
        INIT_XYZS = np.array(START_POSITIONS)
        TARGET_POS_ARRAY = np.array(TARGET_POSITIONS)
        
        print(f"Target: New randomized positions for {num_drones} drone(s):")
        for i in range(num_drones):
            distance = np.linalg.norm(TARGET_POS_ARRAY[i] - INIT_XYZS[i])
            print(f"   Drone {i}: Start {INIT_XYZS[i]} -> Target {TARGET_POS_ARRAY[i]} (dist: {distance:.2f}m)")
    else:
        # Fixed positions for debugging
        INIT_XYZS = np.array([[0, 0, 1] for i in range(num_drones)])
        TARGET_POS_ARRAY = np.array([[1.5, 1.5, 1.2] for i in range(num_drones)])
        print(f"🔧 Using conservative fixed positions for {num_drones} drone(s)")
    
    return INIT_XYZS, TARGET_POS_ARRAY

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
        trajectory=DEFAULT_TRAJECTORY,
        custom_pid_params=None,  # Optional custom PID parameters
        use_best_params=True,    # Use recommended conservative parameters by default
        best_params_file='Best_PID_Params/recommended_hover.json',
        randomize_positions=True,  # Enable randomization of start/target positions
        max_tilt_angle_deg=45,     # Maximum tilt angle in degrees (allows reasonable PID maneuvering)
        max_motor_output_pct=80,    # Maximum motor output percentage (reduced from 85)
        # Episode management parameters
        max_episodes=DEFAULT_MAX_EPISODES,
        episode_timeout_sec=DEFAULT_EPISODE_TIMEOUT_SEC,
        target_hover_time=DEFAULT_TARGET_HOVER_TIME,
        position_threshold=DEFAULT_POSITION_THRESHOLD,
        crash_altitude=DEFAULT_CRASH_ALTITUDE,
        # Performance evaluation parameters
        enable_performance_eval=False,  # Enable detailed performance evaluation
        performance_output_folder=None,  # Custom output folder for performance results
        # Noise parameters
        enable_noise=False,
        noise_level='medium',
        noise_decay=True
        ):
    
    # Generate initial positions for first episode
    INIT_XYZS, TARGET_POS_ARRAY = generate_episode_positions(num_drones, randomize_positions)

    #### Initialize orientation ################################
    INIT_RPYS = np.array([[0, 0, i * (np.pi/2)/num_drones] for i in range(num_drones)])
    
    # Hover mode - target positions are constant
    print(f"🚁 Hover mode: Drones will navigate to target and hover")

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=False,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Visualize targets and paths (only if GUI is enabled) ####
    target_visual_ids = []
    connection_line_ids = []
    
    if gui:
        print("Target: Creating target visualizations...")
        
        # 创建目标可视化
        target_visual_ids = visualize_targets(TARGET_POS_ARRAY, PYB_CLIENT, target_radius=0.15)
        
        # 绘制从起始位置到目标的连接线
        connection_line_ids = draw_connection_lines(INIT_XYZS, TARGET_POS_ARRAY, PYB_CLIENT)
        
        print(f"Success: Created {len(target_visual_ids)//2} target visualizations and {len(connection_line_ids)} connection lines")

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize noise manager ##############################
    noise_manager = None
    if enable_noise:
        if noise_level == 'light':
            noise_manager = create_light_noise_manager()
        elif noise_level == 'heavy':
            noise_manager = create_heavy_noise_manager()
        else: # medium
            noise_manager = GaussianNoiseManager()
        
        # Configure decay
        for config in noise_manager.noise_configs.values():
            config.adaptive = noise_decay

        print("🔊 Gaussian noise enabled.")
    elif enable_noise:
        print("Warning: Noise manager not available, noise will not be added.")
    

    #### Initialize performance evaluator (if enabled) ######
    performance_evaluator = None
    if enable_performance_eval:
        # DronePerformanceLogger will create a 'performance_analysis' subdirectory
        # So we pass the base output_folder directly to avoid double nesting
        perf_output = performance_output_folder if performance_output_folder else output_folder
        performance_evaluator = DronePerformanceLogger(
            output_folder=perf_output, 
            logging_freq=control_freq_hz,
            num_drones=num_drones,
            duration_sec=episode_timeout_sec
        )
        print(f"Stats: Performance evaluation enabled - output: {performance_evaluator.performance_dir}")
        
        # Set target position for evaluation
        target_pos_for_eval = TARGET_POS_ARRAY[0] if len(TARGET_POS_ARRAY) > 0 else np.array([1.5, 1.5, 1.2])
        performance_evaluator.target_position = target_pos_for_eval
    elif enable_performance_eval:
        print("Warning:  Performance evaluation requested but not available")
    
    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
            if use_best_params and custom_pid_params is None:
                custom_pid_params = load_best_pid_params()
            if custom_pid_params is not None:
                # 使用自定义PID参数
                ctrl = []
                for i in range(num_drones):
                    print(f"Safety: Using DSLPIDControl with safety features for drone {i}")
                    controller = DSLPIDControl(drone_model=drone)
                    
                    # 可选：调整安全参数
                    if hasattr(controller, 'MAX_TILT_ANGLE'):
                        controller.MAX_TILT_ANGLE = np.radians(max_tilt_angle_deg)
                    if hasattr(controller, 'MAX_MOTOR_OUTPUT_FRACTION'):
                        controller.MAX_MOTOR_OUTPUT_FRACTION = max_motor_output_pct/100.0
                        controller.SAFE_MAX_PWM = int(controller.MAX_PWM * controller.MAX_MOTOR_OUTPUT_FRACTION)
                    
                    # 设置位置控制PID参数
                    controller.P_COEFF_FOR = np.array(custom_pid_params['pos_p'])
                    controller.I_COEFF_FOR = np.array(custom_pid_params['pos_i'])
                    controller.D_COEFF_FOR = np.array(custom_pid_params['pos_d'])
                    
                    # 设置姿态控制PID参数
                    controller.P_COEFF_TOR = np.array(custom_pid_params['att_p'])
                    controller.I_COEFF_TOR = np.array(custom_pid_params['att_i'])
                    controller.D_COEFF_TOR = np.array(custom_pid_params['att_d'])
                    
                    ctrl.append(controller)
            else:
                # 使用默认参数
                print(f"Safety: Using DSLPIDControl with safety features and default parameters")
                ctrl = []
                for i in range(num_drones):
                    controller = DSLPIDControl(drone_model=drone)
                    
                    # 可选：调整安全参数
                    if hasattr(controller, 'MAX_TILT_ANGLE'):
                        controller.MAX_TILT_ANGLE = np.radians(max_tilt_angle_deg)
                    if hasattr(controller, 'MAX_MOTOR_OUTPUT_FRACTION'):
                        controller.MAX_MOTOR_OUTPUT_FRACTION = max_motor_output_pct/100.0
                        controller.SAFE_MAX_PWM = int(controller.MAX_PWM * controller.MAX_MOTOR_OUTPUT_FRACTION)
                    
                    ctrl.append(controller)

            # Print controller status
            if hasattr(ctrl[0], 'getSafetyStatus'):
                status = ctrl[0].getSafetyStatus()
                print(f"Safety: Safety Controller Status:")
                print(f"   Max tilt angle: {status['max_tilt_angle_deg']:.1f}°")
                print(f"   Max motor output: {status['max_motor_output_pct']:.1f}%")
                print(f"   Safe PWM limit: {status['safe_max_pwm']}")
                print(f"   Startup duration: {status['startup_duration']:.1f}s")

            print(f"Position PID - P: {ctrl[0].P_COEFF_FOR}")
            print(f"Position PID - I: {ctrl[0].I_COEFF_FOR}")
            print(f"Position PID - D: {ctrl[0].D_COEFF_FOR}")
            print(f"Attitude PID - P: {ctrl[0].P_COEFF_TOR}")
            print(f"Attitude PID - I: {ctrl[0].I_COEFF_TOR}")
            print(f"Attitude PID - D: {ctrl[0].D_COEFF_TOR}")

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    
    # Episode management variables
    total_episodes = 0
    successful_episodes = 0
    episode_data = []
    
    print(f"\nStarting: Starting multi-episode simulation:")
    print(f"   Max episodes: {max_episodes}")
    print(f"   Episode timeout: {episode_timeout_sec}s")
    print(f"   Hover time required: {target_hover_time}s")
    print(f"   Position threshold: {position_threshold}m")
    print(f"   Success criteria: Hover at target for {target_hover_time}s")
    
    while total_episodes < max_episodes:
        # Generate new positions for each episode
        if total_episodes > 0:  # Skip first episode as positions are already generated
            INIT_XYZS, TARGET_POS_ARRAY = generate_episode_positions(num_drones, randomize_positions)
        
        # Start new episode
        episode_start_time = time.time()
        episode_step_count = 0
        hover_start_time = None
        hover_duration = 0.0
        episode_success = False
        termination_reason = "timeout"
        
        print(f"\nStats: Episode {total_episodes + 1}/{max_episodes} starting...")
        
        # Start performance evaluation episode
        if performance_evaluator:
            # Update target position for performance evaluator
            target_pos_for_eval = TARGET_POS_ARRAY[0] if len(TARGET_POS_ARRAY) > 0 else np.array([1.5, 1.5, 1.2])
            performance_evaluator.start_episode()
        
        # Reset environment with new positions for new episode
        env.INIT_XYZS = INIT_XYZS
        env.INIT_RPYS = np.array([[0, 0, i * (np.pi/2)/num_drones] for i in range(num_drones)])
        obs, info = env.reset()
        
        # Reset PID controllers for new episode to prevent integral windup
        if drone in [DroneModel.CF2X, DroneModel.CF2P]:
            for i in range(num_drones):
                ctrl[i].reset()  # Reset PID controller state
                print(f"🔄 Reset PID controller for drone {i}")
        
        # Re-create target visualizations for each episode (if GUI is enabled)
        if gui:
            # Clean up previous visualizations
            for visual_id in target_visual_ids:
                try:
                    p.removeBody(visual_id, physicsClientId=PYB_CLIENT)
                except:
                    pass  # Ignore already deleted objects
            for line_id in connection_line_ids:
                try:
                    p.removeUserDebugItem(line_id, physicsClientId=PYB_CLIENT)
                except:
                    pass
            
            # Re-create visualizations with new positions
            target_visual_ids = visualize_targets(TARGET_POS_ARRAY, PYB_CLIENT, target_radius=0.15)
            connection_line_ids = draw_connection_lines(INIT_XYZS, TARGET_POS_ARRAY, PYB_CLIENT)
        
        # Episode simulation loop
        for i in range(0, int(episode_timeout_sec * env.CTRL_FREQ)):
            current_time = i / env.CTRL_FREQ
            episode_step_count += 1

            #### Step the simulation ###################################
            obs, reward, terminated, truncated, info = env.step(action)

            # Add Gaussian noise if enabled
            if noise_manager:
                for i in range(num_drones):
                    obs[i] = noise_manager.add_observation_noise(obs[i])
                noise_manager.step()

            # Update hover timer and check termination conditions
            hover_duration, is_hovering = update_hover_timer(obs, TARGET_POS_ARRAY, hover_start_time, 
                                                            current_time, position_threshold)
            
            # Update hover_start_time based on hovering state
            if is_hovering and hover_start_time is None:
                # Just started hovering
                hover_start_time = current_time
                hover_duration = 0.0  # Reset duration since we just started
            elif not is_hovering:
                # Not hovering, reset timer
                hover_start_time = None
                hover_duration = 0.0
            
            episode_terminated, reason = check_episode_termination(
                obs, current_time, hover_duration, episode_timeout_sec, 
                target_hover_time, crash_altitude
            )
            
            if episode_terminated:
                termination_reason = reason
                episode_success = (reason == "success")
                break

            #### Compute control for the current target position #####
            for j in range(num_drones):
                action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                        state=obs[j],
                                                                        target_pos=TARGET_POS_ARRAY[j],  # Fixed target for hovering
                                                                        target_rpy=INIT_RPYS[j, :]
                                                                        )
            
            # Log performance data (if evaluator is enabled)
            if performance_evaluator:
                # Create reward and info dictionaries for compatibility
                reward = 1.0 if hover_duration > 0 else 0.0  # Simple reward based on hover
                info_dict = {'target_pos': TARGET_POS_ARRAY[0]}
                
                # Create 12-dimensional control array (pos_x_target, pos_y_target, pos_z_target, vel_x_target, vel_y_target, vel_z_target,
                # roll_target, pitch_target, yaw_target, p_roll_target, p_pitch_target, p_yaw_target)
                control_12d = np.zeros(12)
                control_12d[0:3] = TARGET_POS_ARRAY[0]  # Position targets
                # Leave velocity, attitude, and angular velocity targets as zeros
                
                performance_evaluator.log_step_with_performance(
                    drone=0,
                    timestamp=current_time,
                    state=obs[0],  # First drone
                    action=action[0],  # Motor commands as action
                    reward=reward,
                    info=info_dict,
                    control=control_12d  # 12-dimensional control array
                )

            #### Log the simulation ####################################
            for j in range(num_drones):
                logger.log(drone=j,
                           timestamp=i/env.CTRL_FREQ,
                           state=obs[j],
                           control=np.hstack([TARGET_POS_ARRAY[j], INIT_RPYS[j, :], np.zeros(6)])
                           )

            #### Printout ##############################################
            env.render()

            #### Sync the simulation ###################################
            if gui:
                sync(i, START, env.CTRL_TIMESTEP)
        
        # Episode ended - record statistics
        episode_duration = time.time() - episode_start_time  
        
        # End performance evaluation episode
        if performance_evaluator:
            performance_evaluator.end_episode(drone_id=0)
        
        episode_data.append({
            'episode': total_episodes + 1,
            'success': episode_success,
            'reason': termination_reason,
            'duration': episode_duration,
            'steps': episode_step_count,
            'hover_time': hover_duration
        })
        
        if episode_success:
            successful_episodes += 1
            
        print(f"Stats: Episode {total_episodes + 1} completed:")
        print(f"   Result: {'Success: SUCCESS' if episode_success else 'Error: FAILURE'} ({termination_reason})")
        print(f"   Duration: {episode_duration:.1f}s")
        print(f"   Steps: {episode_step_count}")
        print(f"   Hover time: {hover_duration:.1f}s")
        
        total_episodes += 1

    #### Print multi-episode summary ############################
    simulation_duration = time.time() - START
    print(f"\n🏆 Multi-episode simulation completed!")
    print(f"   Total episodes: {total_episodes}")
    print(f"   Successful episodes: {successful_episodes}")
    print(f"   Success rate: {successful_episodes/total_episodes*100:.1f}%")
    print(f"   Total simulation time: {simulation_duration:.1f}s")
    
    if episode_data:
        avg_duration = np.mean([ep['duration'] for ep in episode_data])
        avg_hover_time = np.mean([ep['hover_time'] for ep in episode_data])
        print(f"   Average episode duration: {avg_duration:.1f}s")
        print(f"   Average hover time: {avg_hover_time:.1f}s")
        
        print(f"\n📋 Episode breakdown:")
        for ep in episode_data:
            status = "Success:" if ep['success'] else "Error:"
            print(f"   Episode {ep['episode']}: {status} {ep['reason']} ({ep['duration']:.1f}s, {ep['hover_time']:.1f}s hover)")

    #### Clean up visualizations ################################
    if gui and target_visual_ids:
        print("🧹 Cleaning up target visualizations...")
        for visual_id in target_visual_ids:
            try:
                p.removeBody(visual_id, physicsClientId=PYB_CLIENT)
            except:
                pass  # 忽略已经被删除的对象

    #### Close the environment #################################
    env.close()

    #### Finalize performance evaluation ####################
    if performance_evaluator:
        print(f"\nStats: Finalizing performance evaluation...")
        
        # Generate performance visualizations
        performance_evaluator.visualize_performance(save_plots=True)
        
        # Generate performance report
        report_file = performance_evaluator.generate_performance_report()
        
        # Save performance data
        performance_evaluator.save_performance_data(comment="pid_performance_eval")
        
        print(f"Target: Performance evaluation completed!")
        print(f"   Report saved to: {report_file}")
        print(f"   Data saved to: {performance_evaluator.performance_dir}")
    
    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='PID control for hovering at target positions')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2P)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 1)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 30)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "pid_results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--use_best_params',    default=True, type=str2bool,      help='Whether to use best tuned PID parameters (default: True)', metavar='')
    parser.add_argument('--best_params_file',   default='best_pid_params.json', type=str,           help='JSON file containing best PID parameters (default: "best_pid_params.json")', metavar='')
    parser.add_argument('--randomize_positions', default=True, type=str2bool,    help='Whether to randomize start/target positions (default: True)', metavar='')
    parser.add_argument('--max_tilt_angle_deg',  default=45, type=float,         help='Maximum tilt angle in degrees (default: 45)', metavar='')
    parser.add_argument('--max_motor_output_pct', default=80, type=float,        help='Maximum motor output percentage (default: 80)', metavar='')
    # Episode management arguments
    parser.add_argument('--max_episodes',        default=DEFAULT_MAX_EPISODES, type=int,     help='Maximum number of episodes (default: 10)', metavar='')
    parser.add_argument('--episode_timeout_sec', default=DEFAULT_EPISODE_TIMEOUT_SEC, type=int, help='Episode timeout in seconds (default: 60)', metavar='')
    parser.add_argument('--target_hover_time',   default=DEFAULT_TARGET_HOVER_TIME, type=float, help='Required hover time for success (default: 5.0)', metavar='')
    parser.add_argument('--position_threshold',  default=DEFAULT_POSITION_THRESHOLD, type=float, help='Position threshold for hovering (default: 0.1)', metavar='')
    parser.add_argument('--crash_altitude',      default=DEFAULT_CRASH_ALTITUDE, type=float, help='Crash altitude threshold (default: 0.1)', metavar='')
    # Performance evaluation arguments
    parser.add_argument('--enable_performance_eval', type=str2bool, default=False, help='Enable detailed performance evaluation')
    parser.add_argument('--performance_output_folder', type=str, default=None, help='Custom output folder for performance results')
    
    # Gaussian noise parameters
    parser.add_argument('--enable_noise', default=False, type=str2bool,
                       help='Whether to enable Gaussian noise during training (default: False)')
    parser.add_argument('--noise_level', default='medium', type=str,
                       choices=['light', 'medium', 'heavy'],
                       help='Noise level: light, medium, or heavy (default: medium)')
    parser.add_argument('--noise_decay', default=True, type=str2bool,
                       help='Whether noise should decay with training progress (default: True)')

    ARGS = parser.parse_args()

    run(**vars(ARGS))
