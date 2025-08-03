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

"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Parameters
----------
--drone : DroneModel
    Type of drone model to use (default: CF2X)
--num_drones : int
    Number of drones in the simulation (default: 3)
--physics : Physics
    Physics engine to use (default: PyBullet)
--gui : bool
    Whether to use PyBullet GUI (default: True)
--record_video : bool
    Whether to record video of the simulation (default: False)
--plot : bool
    Whether to plot the simulation results (default: True)
--user_debug_gui : bool
    Whether to add debug lines and parameters to GUI (default: False)
--obstacles : bool
    Whether to add obstacles to the environment (default: True)
--simulation_freq_hz : int
    Simulation frequency in Hz (default: 240)
--control_freq_hz : int
    Control frequency in Hz (default: 48)
--duration_sec : int
    Duration of the simulation in seconds (default: 12)
--output_folder : str
    Folder where to save logs (default: 'results')

Features
--------
- Multi-drone simulation with independent PID controllers
- Circular trajectory following at different altitudes
- Visualization options via PyBullet GUI
- Data logging and plotting capabilities
- Customizable simulation parameters

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

Outputs
-------
- Simulation logs saved to the specified output folder
- Optional CSV export of flight data
- Optional plots showing drone positions, orientations, and control inputs
"""

import json
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2p")
DEFAULT_NUM_DRONES = 5
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 30
DEFAULT_OUTPUT_FOLDER = 'pid_results'
DEFAULT_COLAB = False
DEFAULT_TRAJECTORY = "circle"  # "circle", "figure8", "hover"

def load_best_pid_params(json_file="Best_PID_Params/best_pid_params.json"):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir)
        
        json_path = os.path.join(project_root, json_file)
        json_path = os.path.normpath(json_path)  # 规范化路径
        
        print(f"🔍 Looking for PID parameters at: {json_path}")
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            best_params = data['best_params']
            print(f"🏆 Loading best PID parameters (Score: {data['best_score']:.4f})")
            print(f"📅 Generated: {data.get('timestamp', 'Unknown')}")
            print(f"🎯 Parameter set: {best_params['name']}")
            
            return best_params
        else:
            print(f"❌ Best PID parameters file not found: {json_path}")
            print(f"💡 Make sure the file exists in the project root directory")
            return None
    except Exception as e:
        print(f"❌ Failed to load PID parameters: {e}")
        return None
    
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
        use_best_params=True,
        best_params_file='Best_PID_Params/best_pid_params.json'
        ):
    
    #### Initialize the simulation #############################
    INIT_H = 1.0
    TARGET_H = 0.5
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, INIT_H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])
    
    PERIOD = 10
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))

    #### Initialize a circular trajectory ######################
    if trajectory == "circle":
        for i in range(NUM_WP):
            TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    elif trajectory == "figure8":
        a=R # radius of the figure 8
        for i in range(NUM_WP):
            t = (i/NUM_WP)*2*np.pi
            x = a*np.sin(t)/(1+np.cos(t)**2)
            y = a*np.sin(t)*np.cos(t)/(1+np.cos(t)**2)
            TARGET_POS[i, :] = x+INIT_XYZS[0, 0], y+INIT_XYZS[0, 1], 0

    elif trajectory == "hover":
        for i in range(NUM_WP):
            TARGET_POS[i, :] = INIT_XYZS[0, 0], INIT_XYZS[0, 1], 0

    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

    #### Debug trajectory ######################################
    #### Uncomment alt. target_pos in .computeControlFromState()
    # INIT_XYZS = np.array([[.3 * i, 0, .1] for i in range(num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/3)/num_drones] for i in range(num_drones)])
    # NUM_WP = control_freq_hz*15
    # TARGET_POS = np.zeros((NUM_WP,3))
    # for i in range(NUM_WP):
    #     if i < NUM_WP/6:
    #         TARGET_POS[i, :] = (i*6)/NUM_WP, 0, 0.5*(i*6)/NUM_WP
    #     elif i < 2 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-NUM_WP/6)*6)/NUM_WP, 0, 0.5 - 0.5*((i-NUM_WP/6)*6)/NUM_WP
    #     elif i < 3 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, ((i-2*NUM_WP/6)*6)/NUM_WP, 0.5*((i-2*NUM_WP/6)*6)/NUM_WP
    #     elif i < 4 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, 1 - ((i-3*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-3*NUM_WP/6)*6)/NUM_WP
    #     elif i < 5 * NUM_WP/6:
    #         TARGET_POS[i, :] = ((i-4*NUM_WP/6)*6)/NUM_WP, ((i-4*NUM_WP/6)*6)/NUM_WP, 0.5*((i-4*NUM_WP/6)*6)/NUM_WP
    #     elif i < 6 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-5*NUM_WP/6)*6)/NUM_WP
    # wp_counters = np.array([0 for i in range(num_drones)])

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

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
            if use_best_params and custom_pid_params is None:
                custom_pid_params = load_best_pid_params()
            if custom_pid_params is not None:
                # 使用自定义PID参数
                ctrl = []
                for i in range(num_drones):
                    controller = DSLPIDControl(drone_model=drone)
                    
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
                ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]


            print(f"Position PID - P: {ctrl[0].P_COEFF_FOR}")
            print(f"Position PID - I: {ctrl[0].I_COEFF_FOR}")
            print(f"Position PID - D: {ctrl[0].D_COEFF_FOR}")
            print(f"Attitude PID - P: {ctrl[0].P_COEFF_TOR}")
            print(f"Attitude PID - I: {ctrl[0].I_COEFF_TOR}")
            print(f"Attitude PID - D: {ctrl[0].D_COEFF_TOR}")

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], TARGET_H + j*H_STEP]),# 设置目标高度
                                                                    # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                    target_rpy=INIT_RPYS[j, :]
                                                                    )

        #### Go to the next way point and loop #####################
        for j in range(num_drones):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--trajectory',          default=DEFAULT_TRAJECTORY, type=str,           help='Trajectory to follow (default: "circle")', metavar='', choices=["circle", "figure8"])
    parser.add_argument('--use_best_params',    default=False, type=str2bool,      help='Whether to use best tuned PID parameters (default: False)', metavar='')
    parser.add_argument('--best_params_file',   default='best_pid_params.json', type=str,           help='JSON file containing best PID parameters (default: "best_pid_params.json")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
