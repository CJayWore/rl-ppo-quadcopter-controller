"""
Deep Reinforcement Learning Environment for Drone Navigation.

This module implements a comprehensive drone control environment that combines:
- Path planning and navigation to target positions
- Obstacle avoidance capabilities  
- Precision hovering at target locations
- Gaussian noise simulation for realistic training
- Advanced camera control systems

The environment supports multiple task configurations and provides detailed
performance metrics for training evaluation.

Classes:
    DRLAviary: Main environment class for drone navigation tasks
"""

import dis
import numpy as np
import pybullet as p
import time
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

# Import Gaussian Noise functionality
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'rl_framework'))
    from gaussian_noise import GaussianNoiseManager, NoiseType
except ImportError:
    print("Warning: Gaussian Noise functionality not available - noise parameters will be ignored")
    GaussianNoiseManager = None
    NoiseType = None

class DRLAviary(BaseRLAviary):
    """
    Deep Reinforcement Learning environment for drone navigation tasks.
    
    This environment combines three main capabilities:
    1. Path planning: Navigate towards a target position
    2. Obstacle avoidance: Avoid obstacles along the path  
    3. Hovering: Maintain stable hovering at the target position
    
    The environment supports configurable difficulty through obstacle density,
    noise simulation, and target precision requirements.
    
    Attributes:
        RANDOMIZE_INIT (bool): Whether to randomize initial positions
        NUM_OBSTACLES (int): Number of obstacles in the environment
        OBSTACLE_RADIUS (float): Radius of obstacles
        SENSING_RANGE (float): Maximum sensor detection range
        TARGET_RADIUS (float): Target area radius for completion
        HOVER_THRESHOLD (float): Distance threshold for hovering
        EPISODE_LEN_SEC (int): Maximum episode length in seconds
        ENABLE_OBSTACLES (bool): Whether obstacles are enabled
    """

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 randomize_init: bool = True,
                 num_obstacles: int = 8,
                 obstacle_radius: float = 0.25,
                 sensing_range: float = 2.0,
                 target_radius: float = 0.15,
                 hover_threshold: float = 0.5,
                 episode_len_sec: int = 30,
                 enable_obstacles: bool = True,
                 # Gaussian Noise parameters
                 enable_noise: bool = False,
                 noise_level: str = "medium",  # "light", "medium", "heavy"
                 noise_decay: bool = True,
                 show_noise_ui: bool = True):
        """
        Initialize the DRL environment with specified parameters.
        
        Args:
            drone_model: Type of drone model to simulate
            initial_xyzs: Initial drone positions
            initial_rpys: Initial drone orientations
            physics: Physics engine type
            pyb_freq: PyBullet simulation frequency
            ctrl_freq: Control frequency
            gui: Whether to show GUI
            record: Whether to record simulation
            obs: Observation space type
            act: Action space type
            randomize_init: Whether to randomize initial positions
            num_obstacles: Number of obstacles in environment
            obstacle_radius: Radius of each obstacle
            sensing_range: Maximum sensor detection range
            target_radius: Radius of target completion area
            hover_threshold: Distance threshold for hovering detection
            episode_len_sec: Maximum episode length in seconds
            enable_obstacles: Whether to enable obstacles
            enable_noise: Whether to enable Gaussian noise simulation
            noise_level: Noise intensity level ("light", "medium", "heavy")
            noise_decay: Whether noise should decay during training
            show_noise_ui: Whether to display noise parameters in GUI
        """
        
        # Task configuration
        self.RANDOMIZE_INIT = randomize_init
        self.NUM_OBSTACLES = num_obstacles
        self.OBSTACLE_RADIUS = obstacle_radius
        self.SENSING_RANGE = sensing_range
        self.TARGET_RADIUS = target_radius
        self.HOVER_THRESHOLD = hover_threshold
        self.EPISODE_LEN_SEC = episode_len_sec
        self.ENABLE_OBSTACLES = enable_obstacles

        # Task state
        self.START_POS = np.array([0, 0, 1])
        self.TARGET_POS = np.array([3, 3, 1.5])
        self.obstacle_positions = []
        self.obstacle_ids = []
        
        # Performance metrics
        self.time_at_target = 0
        self.required_hover_time = 5.0
        self.navigation_progress = 0.0
        self.last_distance_to_target = 0.0

        # Timing metrics
        self.episode_start_time = None
        self.episode_count = 0
        self.episode_times = []
        self.total_training_time = 0.0
        self.task_completion_times = []
        
        # Visualization
        self.target_visual_id = None
        self.connection_line_id = None
        self.path_markers = []

        # Gaussian Noise parameters
        self.enable_noise = enable_noise
        self.noise_level = noise_level
        self.noise_decay = noise_decay
        self.show_noise_ui = show_noise_ui
        self.total_steps = 0
        self.total_episodes = 0
        
        # Initialize noise manager
        self.noise_manager = None
        if self.enable_noise and GaussianNoiseManager is not None:
            self.noise_manager = self._setup_noise_manager()
            print(f"Noise-enhanced environment initialized (level: {noise_level})")
        elif self.enable_noise and GaussianNoiseManager is None:
            print("Warning: Noise requested but GaussianNoiseManager not available")
        else:
            print("Noise-free environment initialized")

        # Camera settings
        self.camera_follow_enabled = True
        self.camera_mode = "chase"  # "follow", "chase", "orbit", "target_center", "target_orbit"
        self.camera_distance = 4.0
        self.camera_height_offset = 1.5
        
        # Camera smoothing parameters
        self.camera_smoothing = 0.05  # Smoothing factor, smaller = smoother
        self.camera_update_freq = 8
        self.last_camera_pos = None
        self.last_camera_target = None
        self.last_camera_yaw = 0
        self.last_camera_pitch = -25

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=np.array([self.START_POS]),
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)
        
        self.target_body_ids = []
        self._setupObservationSpace()  # Setup extended observation space

    def _setupObservationSpace(self):
        """Setup extended observation space with additional sensor information."""
        # Get original observation space dimensions
        original_obs_space = super()._observationSpace()
    
        if len(original_obs_space.shape) > 1:
            # If shape is (NUM_DRONES, obs_dim)
            print(f"[DRLAviary] Original obs space shape: {original_obs_space.shape}")
            original_obs_dim = original_obs_space.shape[1]
        else:
            print(f"[DRLAviary] Modified obs space shape: {original_obs_space.shape}")
            # If shape is (obs_dim,)
            original_obs_dim = original_obs_space.shape[0]
        
        num_lidar_rays = 12   
        target_info_world = 3
        target_info_body = 3
        velocity_body = 3
        distance_info = 1
        angle_info = 3
        
        new_obs_dim = original_obs_dim + num_lidar_rays + target_info_world + target_info_body + velocity_body + distance_info + angle_info
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(new_obs_dim,), 
            dtype=np.float32
        )
        
        print(f"[DRLAviary] Original obs dim: {original_obs_dim}")
        print(f"[DRLAviary] New obs dim: {new_obs_dim}")
        print(f"[DRLAviary] Added: {num_lidar_rays} lidar + {target_info_world} world_target + {target_info_body} body_target + {velocity_body} body_vel + {distance_info + angle_info} dist_angle")

    def _setup_noise_manager(self):
        """Create noise manager based on specified noise level."""
        def get_training_progress():
            return min(1.0, self.total_episodes / 1000.0)
        
        manager = GaussianNoiseManager(
            training_progress_callback=get_training_progress,
            random_seed=None
        )
        
        # Setup UI display if supported
        if self.show_noise_ui and self.GUI and hasattr(manager, '_setup_ui_display'):
            try:
                manager.client_id = getattr(self, 'CLIENT', 0)
                manager._setup_ui_display()
            except Exception as e:
                print(f"Warning: Could not setup noise UI: {e}")
        
        # Adjust parameters based on noise level
        if self.noise_level == "light":
            for noise_type in manager.noise_configs:
                manager.noise_configs[noise_type].std_dev *= 0.5
        elif self.noise_level == "heavy":
            for noise_type in manager.noise_configs:
                manager.noise_configs[noise_type].std_dev *= 1.5
        # "medium" level keeps default settings
        
        # Configure noise decay
        if not self.noise_decay:
            for noise_type in manager.noise_configs:
                manager.noise_configs[noise_type].decay_rate = 0.0
        
        return manager

    def _get_lidar_readings(self, drone_pos):
        """Get 12-directional LiDAR readings for obstacle detection."""
        num_rays = 12
        max_distance = self.SENSING_RANGE
        
        distances = []
        for i in range(num_rays):
            # Horizontal 8 rays
            if i < 8:
                angle = 2 * np.pi * i / 8
                direction = np.array([np.cos(angle), np.sin(angle), 0])
            # Vertical 4 rays
            else:
                angles = [-np.pi/4, np.pi/4, -np.pi/6, np.pi/6]
                angle = angles[i-8]
                direction = np.array([np.cos(angle), 0, np.sin(angle)])  # Forward tilt
            
            ray_start = drone_pos
            ray_end = drone_pos + direction * max_distance
            
            hit_info = p.rayTest(ray_start, ray_end, physicsClientId=self.CLIENT)
            
            if hit_info[0][0] != -1:
                hit_body_id = hit_info[0][0]
                
                # Filter out target point body IDs
                if hit_body_id in self.target_body_ids:
                    distances.append(max_distance)  # Ignore target, treat as no obstacle
                else:
                    hit_distance = hit_info[0][2] * max_distance
                    distances.append(hit_distance)
            else:
                distances.append(max_distance)
        
        return np.array(distances)

    def _update_hover_timer(self, drone_pos):
        """Update hovering timer considering both position and angular velocity stability."""
        state = self._getDroneStateVector(0)
        angular_vel = state[13:16]
        angular_velocity_norm = np.linalg.norm(angular_vel)
        
        distance_to_target = np.linalg.norm(self.TARGET_POS - drone_pos)
        
        # Valid hovering requires both position and angular velocity conditions
        if (distance_to_target < self.HOVER_THRESHOLD and 
            angular_velocity_norm < 0.3):  # Angular velocity threshold
            self.time_at_target += 1.0 / self.CTRL_FREQ
        else:
            self.time_at_target = 0  # Reset when conditions not met

    def _computeObs(self):
        """Override to include LiDAR and target information in the observation."""
        # Get base observation
        base_obs = super()._computeObs()
        
        # Ensure base_obs is 1D array
        if isinstance(base_obs, np.ndarray):
            if base_obs.ndim > 1:
                base_obs = base_obs.flatten()
        else:
            base_obs = np.array(base_obs).flatten()
        
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        rpy = state[7:10]
        current_vel = state[10:13]
        angular_vel = state[13:16]
        
        # Apply sensor noise to state readings if noise is enabled
        if self.enable_noise and self.noise_manager is not None:
            # Apply sensor noise to position, velocity, and angular velocity
            current_pos_noisy, current_vel_noisy, angular_vel_noisy = self.noise_manager.add_sensor_noise(
                current_pos, current_vel, angular_vel
            )
            # Apply attitude noise
            rpy_noisy = self.noise_manager.add_attitude_noise(rpy)
            # Apply wind disturbance
            current_vel_noisy = self.noise_manager.add_wind_disturbance(current_vel_noisy)
            
            # Use noisy values for calculations
            current_pos = current_pos_noisy
            current_vel = current_vel_noisy
            rpy = rpy_noisy
        
        lidar_readings = self._get_lidar_readings(current_pos)
        relative_target = self.TARGET_POS - current_pos
        
        # Body frame relative target position
        # Convert world frame relative target position to body frame
        yaw = rpy[2]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        
        # Rotation matrix (only yaw considered, as roll and pitch mainly control movement)
        relative_target_body = np.array([
            cos_yaw * relative_target[0] + sin_yaw * relative_target[1],  # Forward-backward direction
            -sin_yaw * relative_target[0] + cos_yaw * relative_target[1], # Left-right direction
            relative_target[2]  # Up-down direction
        ])
        
        # Body frame velocity components
        velocity_body = np.array([
            cos_yaw * current_vel[0] + sin_yaw * current_vel[1],  # Forward-backward velocity
            -sin_yaw * current_vel[0] + cos_yaw * current_vel[1], # Left-right velocity
            current_vel[2]  # Up-down velocity
        ])
        
        # Distance and direction information to target
        distance_to_target = np.linalg.norm(relative_target)
        target_angle = np.arctan2(relative_target[1], relative_target[0]) - yaw
        # Normalize angle to [-π, π]
        while target_angle > np.pi:
            target_angle -= 2 * np.pi
        while target_angle < -np.pi:
            target_angle += 2 * np.pi
        
        enhanced_obs = np.concatenate([
            base_obs,                    # Original observation
            lidar_readings,              # 12 LiDAR readings
            relative_target,             # World frame relative target position (3)
            relative_target_body,        # Body frame relative target position (3)
            velocity_body,               # Body frame velocity (3)
            [distance_to_target],        # Distance to target (1)
            [target_angle],              # Target angle (1)
            [np.sin(target_angle), np.cos(target_angle)]  # Angle sin/cos representation (2)
        ])
        
        # Apply observation noise if enabled
        if self.enable_noise and self.noise_manager is not None:
            enhanced_obs = self.noise_manager.add_observation_noise(enhanced_obs)
        
        return enhanced_obs.astype(np.float32)
    
    ########################################################################
    # Pybullet methods
    def _drawConnectionLine(self):
        """Draw connection line between drone and target with status-based color coding."""
        if self.GUI:
            drone_state = self._getDroneStateVector(0)
            drone_pos = drone_state[0:3]
            
            if self.connection_line_id is not None:
                p.removeUserDebugItem(self.connection_line_id, physicsClientId=self.CLIENT)
            
            # Choose color based on task status
            if self.time_at_target >= self.required_hover_time:
                color = [0, 1, 0]  # Green - task completed
            elif self.time_at_target > 0:
                color = [1, 1, 0]  # Yellow - hovering
            else:
                color = [0, 0, 1]  # Blue - navigating
            
            self.connection_line_id = p.addUserDebugLine(
                lineFromXYZ=drone_pos,
                lineToXYZ=self.TARGET_POS,
                lineColorRGB=color,
                lineWidth=3,
                lifeTime=0.2,
                physicsClientId=self.CLIENT
            )

    def _addObstacles(self):
        """Add random obstacles and target visualization to the environment."""
        # Clear existing obstacles
        for obs_id in self.obstacle_ids:
            p.removeBody(obs_id, physicsClientId=self.CLIENT)
        self.obstacle_ids = []
        self.obstacle_positions = []

        # Randomly place obstacles between start and end points
        if self.ENABLE_OBSTACLES:
            for i in range(self.NUM_OBSTACLES):
                max_attempts = 50
                for attempt in range(max_attempts):
                    x = np.random.uniform(-1, 4)
                    y = np.random.uniform(-1, 4)
                    z = np.random.uniform(0.3, 2.0)
                    pos = np.array([x, y, z])
                    
                    # Ensure obstacles are not too close to start and end points
                    start_dist = np.linalg.norm(pos - self.START_POS)
                    target_dist = np.linalg.norm(pos - self.TARGET_POS)
                    
                    # Check if the position is too close to existing obstacles
                    too_close = False
                    for existing_pos in self.obstacle_positions:
                        if np.linalg.norm(pos - existing_pos) < self.OBSTACLE_RADIUS * 3:
                            too_close = True
                            break
                    
                    if (start_dist > 0.8 and target_dist > 0.8 and not too_close):
                        break
                
                # Create obstacles
                obstacle_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=self.OBSTACLE_RADIUS)
                obstacle_visual = p.createVisualShape(p.GEOM_SPHERE, 
                                                    radius=self.OBSTACLE_RADIUS,
                                                    rgbaColor=[0.8, 0.2, 0.2, 0.8])
                
                body_id = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=obstacle_collision,
                                        baseVisualShapeIndex=obstacle_visual,
                                        basePosition=pos,
                                        physicsClientId=self.CLIENT)
                
                self.obstacle_ids.append(body_id)
                self.obstacle_positions.append(pos)

        self._visualizeTarget()

    def _visualizeTarget(self):
        """Visualize the target position and completion area."""
        if self.GUI:
            target_visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.1,
                rgbaColor=[0, 1, 0, 0.3],  # Green semi-transparent
                physicsClientId=self.CLIENT
            )
            
            self.target_visual_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=target_visual,
                basePosition=self.TARGET_POS,
                physicsClientId=self.CLIENT
            )
            
            target_area_visual = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=self.TARGET_RADIUS,
                length=0.02,
                rgbaColor=[0, 1, 0, 0.3],
                physicsClientId=self.CLIENT
            )
            
            self.target_area_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=target_area_visual,
                basePosition=self.TARGET_POS,
                physicsClientId=self.CLIENT
            )

            self.target_body_ids = [self.target_visual_id, self.target_area_id]

    # ======================== Camera Smoothing Helper Functions ========================
    
    def _smooth_interpolate(self, current, target):
        """Smooth interpolation for position vectors."""
        return current + (target - current) * self.camera_smoothing
    
    def _smooth_interpolate_scalar(self, current, target):
        """Smooth interpolation for scalar values."""
        return current + (target - current) * self.camera_smoothing
    
    def _smooth_angle_interpolate(self, current_angle, target_angle):
        """Smooth interpolation for angles (handling angle wrapping)."""
        # Handle cyclical nature of angle differences
        diff = target_angle - current_angle
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        
        return current_angle + diff * self.camera_smoothing

    def _updateCamera(self):
        """Intelligent camera follow system with smooth processing."""
        if not self.GUI or not self.camera_follow_enabled:
            return
        
        # Reduce update frequency to minimize jitter
        if self.step_counter % self.camera_update_freq != 0:
            return
            
        # Get drone state
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        drone_vel = state[10:13]
        
        if self.camera_mode == "follow":
            self._followCamera(drone_pos, drone_vel)
        elif self.camera_mode == "chase":
            self._chaseCamera(drone_pos, drone_vel)
        elif self.camera_mode == "orbit":
            self._orbitCamera(drone_pos)
        elif self.camera_mode == "target_center":
            self._targetCenterCamera(drone_pos)
        elif self.camera_mode == "target_orbit":
            self._targetOrbitCamera(drone_pos)
    
    def _followCamera(self, drone_pos, drone_vel):
        """Third-person follow camera with smooth tracking."""
        # Predict drone position
        predicted_pos = drone_pos + drone_vel * 0.1
        
        # Calculate base camera position
        velocity_direction = drone_vel / (np.linalg.norm(drone_vel) + 1e-6)
        
        # If drone is stationary, use direction towards target
        if np.linalg.norm(drone_vel) < 0.1:
            if hasattr(self, 'TARGET_POS'):
                direction_to_target = self.TARGET_POS - drone_pos
                velocity_direction = direction_to_target / (np.linalg.norm(direction_to_target) + 1e-6)
            else:
                velocity_direction = np.array([1, 0, 0])  # Default direction
        
        # Dynamic camera distance adjustment based on speed and height
        speed = np.linalg.norm(drone_vel)
        height = drone_pos[2]
        
        # Base distance + speed adjustment + height adjustment
        dynamic_distance = self.camera_distance + speed * 0.5 + max(0, (height - 1.0) * 0.3)
        dynamic_distance = max(0.5, min(8.0, dynamic_distance))  # Limit range
        
        # Camera position behind drone
        camera_offset = -velocity_direction * dynamic_distance
        camera_offset[2] += self.camera_height_offset
        
        target_camera_pos = predicted_pos + camera_offset
        target_camera_target = drone_pos + velocity_direction * 2.0
        
        # Initialize history positions
        if self.last_camera_pos is None:
            self.last_camera_pos = target_camera_pos
            self.last_camera_target = target_camera_target
        
        # Smooth interpolation
        smooth_camera_pos = self._smooth_interpolate(self.last_camera_pos, target_camera_pos)
        smooth_camera_target = self._smooth_interpolate(self.last_camera_target, target_camera_target)
        
        # Calculate camera angles
        direction_to_target = smooth_camera_target - smooth_camera_pos
        distance = np.linalg.norm(direction_to_target[:2])
        
        if distance > 1e-6:
            yaw = np.degrees(np.arctan2(direction_to_target[1], direction_to_target[0]))
            pitch = np.degrees(np.arctan2(-direction_to_target[2], distance))
        else:
            yaw, pitch = 0, -30
        
        # Smooth angle processing
        smooth_yaw = self._smooth_angle_interpolate(self.last_camera_yaw, yaw)
        smooth_pitch = self._smooth_interpolate_scalar(self.last_camera_pitch, pitch)
        
        # Apply camera settings
        p.resetDebugVisualizerCamera(
            cameraDistance=dynamic_distance,
            cameraYaw=smooth_yaw,
            cameraPitch=smooth_pitch,
            cameraTargetPosition=drone_pos,
            physicsClientId=self.CLIENT
        )
        
        # Update history values
        self.last_camera_pos = smooth_camera_pos
        self.last_camera_target = smooth_camera_target
        self.last_camera_yaw = smooth_yaw
        self.last_camera_pitch = smooth_pitch

    def _chaseCamera(self, drone_pos, drone_vel):
        """追逐相机（总是从后方跟随）- 平滑版本"""
        # 计算目标角度
        if np.linalg.norm(drone_vel) > 0.1:
            vel_normalized = drone_vel / np.linalg.norm(drone_vel)
            target_yaw = np.degrees(np.arctan2(vel_normalized[1], vel_normalized[0]))
        else:
            # 静止时，面向目标
            if hasattr(self, 'TARGET_POS'):
                direction = self.TARGET_POS - drone_pos
                target_yaw = np.degrees(np.arctan2(direction[1], direction[0]))
            else:
                target_yaw = self.last_camera_yaw  # 保持当前角度
        
        # 🔧 动态相机距离调整 - 根据速度和环境调整
        speed = np.linalg.norm(drone_vel)
        height = drone_pos[2]
        
        # 基础距离 + 速度调整 + 高度调整
        dynamic_distance = self.camera_distance + speed * 0.4 + max(0, (height - 1.0) * 0.2)
        dynamic_distance = max(0.5, min(7.0, dynamic_distance))  # 限制范围
        
        target_pitch = -25  # 稍微向下俯视
        height_offset = 0.5
        
        target_camera_target = drone_pos + np.array([0, 0, height_offset])
        
        # 角度平滑处理
        smooth_yaw = self._smooth_angle_interpolate(self.last_camera_yaw, target_yaw)
        smooth_pitch = self._smooth_interpolate_scalar(self.last_camera_pitch, target_pitch)
        
        # 目标位置平滑处理
        if self.last_camera_target is None:
            self.last_camera_target = target_camera_target
        
        smooth_camera_target = self._smooth_interpolate(self.last_camera_target, target_camera_target)
        
        # 应用相机设置
        p.resetDebugVisualizerCamera(
            cameraDistance=dynamic_distance,
            cameraYaw=smooth_yaw,
            cameraPitch=smooth_pitch,
            cameraTargetPosition=smooth_camera_target,
            physicsClientId=self.CLIENT
        )
        
        # 更新历史值
        self.last_camera_target = smooth_camera_target
        self.last_camera_yaw = smooth_yaw
        self.last_camera_pitch = smooth_pitch

    def _orbitCamera(self, drone_pos):
        """环绕相机（围绕无人机旋转）- 修复版本"""
        # 🔧 修复：使用累加角度而不是取模，避免角度跳跃问题
        orbit_speed = 0.5  # 环绕速度（稍微加快）
        
        # 初始化累积角度
        if not hasattr(self, 'orbit_accumulated_angle'):
            self.orbit_accumulated_angle = self.last_camera_yaw if self.last_camera_yaw is not None else 0
        
        # 直接累加角度，不使用取模
        self.orbit_accumulated_angle += orbit_speed
        
        # 🔧 动态相机距离调整 - 根据无人机高度和环境调整
        height = drone_pos[2]
        
        # 计算到目标的距离，影响相机距离
        drone_to_target_distance = np.linalg.norm(drone_pos - self.TARGET_POS) if hasattr(self, 'TARGET_POS') else 0
        
        # 基础距离 + 高度调整 + 目标距离影响
        dynamic_distance = self.camera_distance * 1.5 + max(0, (height - 1.0) * 0.4) + drone_to_target_distance * 0.1
        dynamic_distance = max(0.5, min(10.0, dynamic_distance))  # 限制范围
        
        target_pitch = -20
        
        target_camera_target = drone_pos + np.array([0, 0, 0.5])
        
        # 🔧 修复：直接使用累积角度，不进行角度插值
        current_yaw = self.orbit_accumulated_angle
        smooth_pitch = self._smooth_interpolate_scalar(self.last_camera_pitch, target_pitch)
        
        # 目标位置平滑处理
        if self.last_camera_target is None:
            self.last_camera_target = target_camera_target
        
        smooth_camera_target = self._smooth_interpolate(self.last_camera_target, target_camera_target)
        
        # 应用相机设置
        p.resetDebugVisualizerCamera(
            cameraDistance=dynamic_distance,
            cameraYaw=current_yaw,
            cameraPitch=smooth_pitch,
            cameraTargetPosition=smooth_camera_target,
            physicsClientId=self.CLIENT
        )
        
        # 更新历史值
        self.last_camera_target = smooth_camera_target
        self.last_camera_yaw = current_yaw  # 使用当前累积角度
        self.last_camera_pitch = smooth_pitch

    def _targetCenterCamera(self, drone_pos):
        """以目标点为中心，始终朝向无人机 - 平滑版本"""
        # 计算从目标点到无人机的方向
        direction_to_drone = drone_pos - self.TARGET_POS
        distance_to_drone = np.linalg.norm(direction_to_drone)
        
        if distance_to_drone < 1e-6:
            # 如果无人机就在目标点，使用默认视角
            target_yaw = 0
            target_pitch = -30
        else:
            # 计算朝向无人机的角度
            target_yaw = np.degrees(np.arctan2(direction_to_drone[1], direction_to_drone[0]))
            # 计算俯仰角（向上看无人机）
            horizontal_distance = np.linalg.norm(direction_to_drone[:2])
            target_pitch = np.degrees(np.arctan2(direction_to_drone[2], horizontal_distance))
        
        # 相机距离设置（从目标点出发）
        camera_distance = max(0.5, distance_to_drone * 0.7)  # 动态调整距离
        target_camera_target = self.TARGET_POS  # 相机始终看向目标点
        
        # 角度平滑处理
        smooth_yaw = self._smooth_angle_interpolate(self.last_camera_yaw, target_yaw)
        smooth_pitch = self._smooth_interpolate_scalar(self.last_camera_pitch, target_pitch)
        
        # 目标位置平滑处理
        if self.last_camera_target is None:
            self.last_camera_target = target_camera_target
        
        smooth_camera_target = self._smooth_interpolate(self.last_camera_target, target_camera_target)
        
        # 应用相机设置
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=smooth_yaw,
            cameraPitch=smooth_pitch,
            cameraTargetPosition=smooth_camera_target,
            physicsClientId=self.CLIENT
        )
        
        # 更新历史值
        self.last_camera_target = smooth_camera_target
        self.last_camera_yaw = smooth_yaw
        self.last_camera_pitch = smooth_pitch

    def _targetOrbitCamera(self, drone_pos):
        """环绕目标点旋转相机 - 修复版本"""
        # 🔧 修复：使用累加角度而不是取模，避免角度跳跃问题
        orbit_speed = 0.5  # 环绕速度
        
        # 初始化累积角度
        if not hasattr(self, 'target_orbit_accumulated_angle'):
            self.target_orbit_accumulated_angle = self.last_camera_yaw if self.last_camera_yaw is not None else 0
        
        # 直接累加角度，不使用取模
        self.target_orbit_accumulated_angle += orbit_speed
        
        # 计算相机距离（基于无人机到目标点的距离动态调整）
        drone_to_target_distance = np.linalg.norm(drone_pos - self.TARGET_POS)
        camera_distance = max(0.5, drone_to_target_distance * 1.2 + 2.0)  # 确保能看到无人机和目标
        
        # 俯仰角略微向下，以便看到目标点
        target_pitch = -15
        
        target_camera_target = self.TARGET_POS  # 相机始终看向目标点
        
        # 🔧 修复：直接使用累积角度，不进行角度插值
        current_yaw = self.target_orbit_accumulated_angle
        smooth_pitch = self._smooth_interpolate_scalar(self.last_camera_pitch, target_pitch)
        
        # 目标位置平滑处理
        if self.last_camera_target is None:
            self.last_camera_target = target_camera_target
        
        smooth_camera_target = self._smooth_interpolate(self.last_camera_target, target_camera_target)
        
        # 应用相机设置
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=current_yaw,
            cameraPitch=smooth_pitch,
            cameraTargetPosition=smooth_camera_target,
            physicsClientId=self.CLIENT
        )
        
        # 更新历史值
        self.last_camera_target = smooth_camera_target
        self.last_camera_yaw = current_yaw
        self.last_camera_pitch = smooth_pitch

    def toggle_camera_mode(self):
        """Toggle camera mode (can be called via keyboard)."""
        modes = ["follow", "chase", "orbit", "target_center", "target_orbit"]
        current_index = modes.index(self.camera_mode)
        self.camera_mode = modes[(current_index + 1) % len(modes)]
        print(f"Camera mode switched to: {self.camera_mode}")

    def set_camera_distance(self, distance):
        """Set camera distance with bounds checking."""
        self.camera_distance = max(0.5, min(10.0, distance))
        print(f"Camera distance set to: {self.camera_distance:.1f}m")

    #########################################################################
    # Reward functions
    def _navigationReward(self):
        """Calculate navigation reward with multi-directional control incentives."""
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        current_vel = state[10:13]
        rpy = state[7:10]
        
        position_error = self.TARGET_POS - current_pos
        current_distance = np.linalg.norm(position_error)

        # Distance-based reward
        distance_reward = min(1000.0, 10.0 / (1e-6 + current_distance))
        
        delta_distance = (self.last_distance_to_target - current_distance)
        approaching_reward = 0
        if delta_distance > 0:
            approaching_reward = delta_distance * 10.0
        self.last_distance_to_target = current_distance

        # Multi-directional control reward
        # Convert position error to body frame
        yaw = rpy[2]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        
        error_body = np.array([
            cos_yaw * position_error[0] + sin_yaw * position_error[1],   # Forward-backward error
            -sin_yaw * position_error[0] + cos_yaw * position_error[1],  # Left-right error
            position_error[2]  # Up-down error
        ])
        
        velocity_body = np.array([
            cos_yaw * current_vel[0] + sin_yaw * current_vel[1],   # Forward-backward velocity
            -sin_yaw * current_vel[0] + cos_yaw * current_vel[1],  # Left-right velocity
            current_vel[2]  # Up-down velocity
        ])
        
        # Multi-directional velocity reward: reward velocity in correct direction
        forward_backward_reward = 0
        left_right_reward = 0
        up_down_reward = 0
        
        # If forward-backward movement needed, reward appropriate velocity
        if abs(error_body[0]) > 0.1:  # Significant forward-backward error
            desired_forward_vel = np.clip(error_body[0] * 2.0, -2.0, 2.0)  # Desired forward-backward velocity
            forward_backward_reward = 5.0 * max(0, 1.0 - abs(velocity_body[0] - desired_forward_vel) / 2.0)
        
        # Left-right direction similar
        if abs(error_body[1]) > 0.1:  # Significant left-right error
            desired_lateral_vel = np.clip(error_body[1] * 2.0, -2.0, 2.0)  # Desired left-right velocity
            left_right_reward = 5.0 * max(0, 1.0 - abs(velocity_body[1] - desired_lateral_vel) / 2.0)
        
        # Up-down direction
        if abs(error_body[2]) > 0.1:  # Significant up-down error
            desired_vertical_vel = np.clip(error_body[2] * 1.5, -1.5, 1.5)  # Desired up-down velocity
            up_down_reward = 5.0 * max(0, 1.0 - abs(velocity_body[2] - desired_vertical_vel) / 1.5)
        
        multi_direction_reward = forward_backward_reward + left_right_reward + up_down_reward

        # Precision reward
        k = 0.3
        x_penalty = 1.0-np.tanh(k * abs(position_error[0]))
        y_penalty = 1.0-np.tanh(k * abs(position_error[1]))
        z_penalty = 1.0-np.tanh(k * abs(position_error[2]))
        position_precision_reward = (x_penalty + y_penalty + z_penalty) * 60

        # Height safety reward
        height_safety_reward = 0
        if current_pos[2] < 0.3:
            height_safety_reward = -5.0 * (0.3 - current_pos[2])
        elif current_pos[2] > 0.5:
            height_safety_reward = 2.0

        # Suppress unnecessary yaw rotation
        yaw_stability_reward = 0
        if current_distance < self.HOVER_THRESHOLD * 2:  # Near target
            angular_vel = state[13:16]
            yaw_angular_vel = abs(angular_vel[2])
            yaw_stability_reward = 3.0 * max(0, 1.0 - yaw_angular_vel / 0.5)

        navigation_reward = (
            distance_reward +
            approaching_reward +
            position_precision_reward +
            height_safety_reward +
            multi_direction_reward +
            yaw_stability_reward
        )
        
        return navigation_reward, {
            'distance_reward': distance_reward,
            'approaching_reward': approaching_reward,
            'position_precision_reward': position_precision_reward,
            'height_safety_reward': height_safety_reward,
            'multi_direction_reward': multi_direction_reward,
            'forward_backward_reward': forward_backward_reward,
            'left_right_reward': left_right_reward,
            'up_down_reward': up_down_reward,
            'yaw_stability_reward': yaw_stability_reward,
            'error_body': error_body,
            'velocity_body': velocity_body,
            'x_penalty': x_penalty,
            'y_penalty': y_penalty,
            'z_penalty': z_penalty
        }
    
    def _stabilityPenalty(self):
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        rpy = state[7:10]
        current_vel = state[10:13]
        angular_vel = state[13:16]
        last_action = state[16:20]

        # 姿态稳定性
        max_tilt = np.pi/12  # 15度作为参考
        roll_stability = max(0, 10.0 * (1.0 - abs(rpy[0]) / max_tilt))
        pitch_stability = max(0, 10.0 * (1.0 - abs(rpy[1]) / max_tilt))

        # 🔧 改进：分别控制各轴角速度，yaw轴在悬停时要求更严格
        distance_to_target = np.linalg.norm(current_pos - self.TARGET_POS)
        
        max_roll_pitch_ang_vel = 1.5
        max_yaw_ang_vel = 1.0 if distance_to_target > self.HOVER_THRESHOLD else 0.3  # 悬停时更严格

        roll_ang_vel_reward = 5.0 * max(0, 1.0 - abs(angular_vel[0]) / max_roll_pitch_ang_vel)
        pitch_ang_vel_reward = 5.0 * max(0, 1.0 - abs(angular_vel[1]) / max_roll_pitch_ang_vel)
        yaw_ang_vel_reward = 8.0 * max(0, 1.0 - abs(angular_vel[2]) / max_yaw_ang_vel)  # yaw权重更高

        angular_velocity_reward = roll_ang_vel_reward + pitch_ang_vel_reward + yaw_ang_vel_reward

        # RPM平滑性
        rpm_smoothness_penalty = -0.1 * np.var(last_action) / (np.mean(last_action) + 1e-6)

        # 🔧 新增：在悬停区域时，额外奖励低yaw角速度
        hover_yaw_bonus = 0
        if distance_to_target < self.HOVER_THRESHOLD:
            hover_yaw_bonus = 5.0 * max(0, 1.0 - abs(angular_vel[2]) / 0.1)  # 非常严格的yaw要求

        # 🔧 新增：yaw稳定性奖励 - 在目标附近时鼓励保持固定yaw
        yaw_stability = 0
        if distance_to_target < self.HOVER_THRESHOLD:
            # 在悬停区域内，强烈鼓励yaw稳定
            if not hasattr(self, 'target_yaw'):
                self.target_yaw = rpy[2]  # 记录第一次进入悬停区域时的yaw
            
            yaw_error = abs(rpy[2] - self.target_yaw)
            # 处理yaw角度的循环性质
            if yaw_error > np.pi:
                yaw_error = 2*np.pi - yaw_error
            
            # yaw稳定性奖励，距离目标越近要求越严格
            yaw_weight = 15.0 * (1.0 - distance_to_target / self.HOVER_THRESHOLD)
            yaw_stability = max(0, yaw_weight * (1.0 - yaw_error / np.pi))

        attitude_penalty = roll_stability + pitch_stability + yaw_stability

        stability_penalty = (
            attitude_penalty +
            angular_velocity_reward +
            hover_yaw_bonus  # 新增
        )

        return stability_penalty, {
            'attitude_penalty': attitude_penalty,
            'roll_stability': roll_stability,
            'pitch_stability': pitch_stability,
            'yaw_stability': yaw_stability,
            'angular_velocity_reward': angular_velocity_reward,
            'roll_ang_vel_reward': roll_ang_vel_reward,
            'pitch_ang_vel_reward': pitch_ang_vel_reward,
            'yaw_ang_vel_reward': yaw_ang_vel_reward,
            'hover_yaw_bonus': hover_yaw_bonus,
            'rpm_smoothness_penalty': rpm_smoothness_penalty,
            'rpy': rpy,
            'angular_vel': angular_vel,
            'distance_to_target': distance_to_target,
            'target_yaw': getattr(self, 'target_yaw', None)
        }

    def _hoveringReward(self):
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        current_vel = state[10:13]
        angular_vel = state[13:16]  # 🔧 添加角速度
        
        distance_to_target = np.linalg.norm(current_pos - self.TARGET_POS)
        velocity_norm = np.linalg.norm(current_vel)
        angular_velocity_norm = np.linalg.norm(angular_vel)  # 🔧 计算角速度模长
        
        ################################################################################################
        MAX_SPEED_REWARD = 100.0  # Max reward for a perfect hover at the target center.
        
        # We set the distance decay so the reward is half its max at the HOVER_THRESHOLD boundary.
        DISTANCE_DECAY = np.log(2) / (self.HOVER_THRESHOLD**2)
        
        # This controls how strongly velocity is penalized. Higher value = more penalty for speed.
        VELOCITY_DECAY = 0.5

        # Calculate a distance-based factor (0 to 1) using a Gaussian function.
        distance_factor = np.exp(-DISTANCE_DECAY * distance_to_target**2)
        
        # Calculate a velocity-based factor (0 to 1) using another Gaussian function.
        velocity_factor = np.exp(-VELOCITY_DECAY * velocity_norm**2)
        
        # 🔧 新增：角速度稳定性因子
        ANGULAR_VELOCITY_DECAY = 1.0  # 角速度衰减参数
        angular_velocity_factor = np.exp(-ANGULAR_VELOCITY_DECAY * angular_velocity_norm**2)
        
        # 🔧 修改：综合考虑线性和角速度
        speed_reward = MAX_SPEED_REWARD * distance_factor * velocity_factor * angular_velocity_factor
        ################################################################################################

        # 悬停时间奖励
        hover_time_reward = 0
        if distance_to_target < self.HOVER_THRESHOLD:
            # 🔧 新增：只有在角速度也足够小时才计算悬停时间
            if angular_velocity_norm < 0.3:  # 角速度阈值
                hover_time_reward = min(20.0, self.time_at_target * 2.0)
            else:
                # 如果在悬停区域但角速度太大，重置悬停时间
                self.time_at_target = max(0, self.time_at_target - 0.1)
        
        hovering_reward = speed_reward + hover_time_reward

        return hovering_reward, {
            'speed_reward': speed_reward,
            'hover_time_reward': hover_time_reward,
            'velocity_norm': velocity_norm,
            'angular_velocity_norm': angular_velocity_norm,
            'distance_to_target': distance_to_target,
            'in_hover_zone': distance_to_target < self.HOVER_THRESHOLD,
            'angular_stable': angular_velocity_norm < 0.3
        }
    
    def _obstacleAvoidanceReward(self):
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        
        obstacle_reward = 0
        if self.ENABLE_OBSTACLES and len(self.obstacle_positions) > 0:
            min_dist_to_obstacle = min([np.linalg.norm(current_pos - obs_pos) for obs_pos in self.obstacle_positions])
            
            ideal_safety_distance = self.OBSTACLE_RADIUS + 0.5
            
            # 使用高斯函数给予一个正向的安全区域奖励
            # 当无人机处于理想距离时，奖励最高
            safety_bonus = 5.0 * np.exp(-((min_dist_to_obstacle - ideal_safety_distance)**2) / (2 * (ideal_safety_distance/2)**2))
            obstacle_reward += safety_bonus

            critical_distance = self.OBSTACLE_RADIUS + 0.3
            for obs_pos in self.obstacle_positions:
                dist_to_obstacle = np.linalg.norm(current_pos - obs_pos)
                if dist_to_obstacle < critical_distance:
                    # 使用指数函数创建强烈的避障信号
                    penalty_factor = np.exp(-(dist_to_obstacle - self.OBSTACLE_RADIUS) * 5.0)
                    obstacle_reward -= 10.0 * penalty_factor
                    
                    # 如果非常接近障碍物，给予额外的强烈惩罚
                    if dist_to_obstacle < self.OBSTACLE_RADIUS + 0.05:
                        obstacle_reward -= 100.0
        
        return obstacle_reward
    def _computeReward(self):
        """Compute total reward from all component rewards."""
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        current_vel = state[10:13]
        current_distance = np.linalg.norm(self.TARGET_POS - current_pos)

        # Calculate reward components
        navigation_reward, nav_details = self._navigationReward()
        stability_reward, stab_details = self._stabilityPenalty()
        hovering_reward, hover_details = self._hoveringReward()
        obstacle_reward = self._obstacleAvoidanceReward()
        
        # Task completion reward
        survival_reward = 0.5
        completion_reward = 0
        if self.time_at_target >= self.required_hover_time:
            completion_reward = 50.0

        total_reward = (
            navigation_reward +
            stability_reward +
            hovering_reward +
            obstacle_reward +
            completion_reward +
            survival_reward
        )

        # Detailed debug output (commented - can be uncommented for debugging)
        if self.step_counter % 100 == 0:
            obstacle_info = f", Obstacles: {len(self.obstacle_positions)}" if self.ENABLE_OBSTACLES else ", No obstacles"
            # print(f"\n--------------------Environment-------------------- "
            #       f"\nTarget: {self.TARGET_POS}, \nDrone Pos: {current_pos}, "
            #       f"\nDistance: {current_distance:.3f},"
            #       f"Speed: {np.linalg.norm(current_vel):.2f}, "
            #       f"Hover time: {self.time_at_target:.1f}s{obstacle_info}")
            
            # print(f"\n--------------------Rewards--------------------")
            # print(f"Navigation ({navigation_reward:.2f}):")
            # print(f"  - Distance: {nav_details['distance_reward']:.2f}")
            # print(f"  - Approaching: {nav_details['approaching_reward']:.2f}")
            # print(f"  - Position Precision: {nav_details['position_precision_reward']:.2f}")
            # print(f"    * X penalty: {nav_details['x_penalty']:.3f}")
            # print(f"    * Y penalty: {nav_details['y_penalty']:.3f}")
            # print(f"    * Z penalty: {nav_details['z_penalty']:.3f}")
            # print(f"  - Multi-direction: {nav_details['multi_direction_reward']:.2f}")
            # print(f"    * Forward/Backward: {nav_details['forward_backward_reward']:.2f}")
            # print(f"    * Left/Right: {nav_details['left_right_reward']:.2f}")
            # print(f"    * Up/Down: {nav_details['up_down_reward']:.2f}")
            # print(f"  - Yaw Stability: {nav_details['yaw_stability_reward']:.2f}")
            
            # print(f"Stability ({stability_reward:.2f}):")
            # print(f"  - Attitude: {stab_details['attitude_penalty']:.2f}")
            # print(f"    * Roll: {stab_details['roll_stability']:.2f}")
            # print(f"    * Pitch: {stab_details['pitch_stability']:.2f}")
            # print(f"    * Yaw: {stab_details['yaw_stability']:.2f} (target: {stab_details['target_yaw']})")
            # print(f"  - Angular Vel: {stab_details['angular_velocity_reward']:.2f}")
            # print(f"    * Yaw ω: {stab_details['yaw_ang_vel_reward']:.2f} (|ωz|: {abs(stab_details['angular_vel'][2]):.3f})")
            # print(f"  - Hover Yaw Bonus: {stab_details['hover_yaw_bonus']:.2f}")

            # print(f"Hovering ({hovering_reward:.2f}):")
            # print(f"  - Speed: {hover_details['speed_reward']:.2f}")
            # print(f"  - Angular stable: {hover_details['angular_stable']}")
            # print(f"  - Hover Time: {hover_details['hover_time_reward']:.2f}")
            
            # print(f"Obstacles ({obstacle_reward:.2f}):")

            
            # print(f"Completion: {completion_reward:.2f}")
            # print(f"TOTAL REWARD: {total_reward:.2f}")
        
        return total_reward

    def _computeTerminated(self):
        """Check if the task is completed (drone has hovered at target for required time)."""
        return self.time_at_target >= self.required_hover_time

    def _computeTruncated(self):
        """Check if the episode should be truncated due to safety violations or time limits."""
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        
        # Obstacle collision detection
        if self.ENABLE_OBSTACLES:
            for obs_pos in self.obstacle_positions:
                if np.linalg.norm(current_pos - obs_pos) < self.OBSTACLE_RADIUS + 0.05:
                    return True
        
        # Boundary violation detection
        if (current_pos[0] < -2 or current_pos[0] > 5 or 
            current_pos[1] < -2 or current_pos[1] > 5 or
            current_pos[2] < 0.05 or current_pos[2] > 3.5):
            return True
        
        # Attitude violation detection
        rpy = state[7:10]
        if abs(rpy[0]) > 1.0 or abs(rpy[1]) > 1.0:  # 57 degrees
            return True
        
        # Time limit detection
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        
        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        
        current_episode_time = time.time() - self.episode_start_time if self.episode_start_time else 0
        
        return {
            "current_pos": current_pos,
            "target_pos": self.TARGET_POS,
            "distance_to_target": np.linalg.norm(current_pos - self.TARGET_POS),
            "time_at_target": self.time_at_target,
            "obstacle_positions": self.obstacle_positions,
            "episode_time": current_episode_time,
            "episode_count": self.episode_count,
            "total_training_time": self.total_training_time,
            "avg_episode_time": np.mean(self.episode_times) if self.episode_times else 0,
            "success_rate": len(self.task_completion_times) / max(1, self.episode_count) * 100,
            "avg_completion_time": np.mean(self.task_completion_times) if self.task_completion_times else 0
        }

    def get_timing_stats(self):
        """获取详细的时间统计信息"""
        if not self.episode_times:
            return {
                "total_episodes": 0,
                "total_training_time": 0,
                "avg_episode_time": 0,
                "min_episode_time": 0,
                "max_episode_time": 0,
                "success_rate": 0,
                "avg_completion_time": 0
            }
        
        return {
            "total_episodes": len(self.episode_times),
            "total_training_time": self.total_training_time,
            "avg_episode_time": np.mean(self.episode_times),
            "min_episode_time": np.min(self.episode_times),
            "max_episode_time": np.max(self.episode_times),
            "std_episode_time": np.std(self.episode_times),
            "success_rate": len(self.task_completion_times) / len(self.episode_times) * 100,
            "avg_completion_time": np.mean(self.task_completion_times) if self.task_completion_times else 0,
            "recent_avg_time": np.mean(self.episode_times[-10:]) if len(self.episode_times) >= 10 else np.mean(self.episode_times)
        }
    
    def print_final_stats(self):
        """打印最终的训练统计信息"""
        stats = self.get_timing_stats()
        
        print("\n" + "="*60)
        print("🏁 FINAL TRAINING STATISTICS")
        print("="*60)
        print(f"Total Episodes: {stats['total_episodes']}")
        print(f"Total Training Time: {stats['total_training_time']/3600:.2f} hours")
        print(f"Average Episode Time: {stats['avg_episode_time']:.2f}s")
        print(f"Min/Max Episode Time: {stats['min_episode_time']:.2f}s / {stats['max_episode_time']:.2f}s")
        print(f"Episode Time Std: {stats['std_episode_time']:.2f}s")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        if stats['avg_completion_time'] > 0:
            print(f"Average Task Completion Time: {stats['avg_completion_time']:.2f}s")
        print(f"Recent Performance (last 10): {stats['recent_avg_time']:.2f}s avg")
        print("="*60)
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        # Update episode count for noise manager
        if self.enable_noise and self.noise_manager is not None:
            self.noise_manager.new_episode()
        
        self.total_episodes += 1
        
        # Record previous episode timing
        if self.episode_start_time is not None:
            episode_duration = time.time() - self.episode_start_time
            self.episode_times.append(episode_duration)
            self.total_training_time += episode_duration
            
            # Print episode statistics
            self.episode_count += 1
            avg_time = np.mean(self.episode_times[-10:]) if len(self.episode_times) >= 10 else np.mean(self.episode_times)
            
            task_completed = self.time_at_target >= self.required_hover_time

            # Essential debug info for performance tracking (commented - can be uncommented)
            # print(f"Episode {self.episode_count} completed:")
            # print(f"   Duration: {episode_duration:.2f}s")
            # print(f"   Hover time: {self.time_at_target:.1f}s")
            # print(f"   Task completed: {'Yes' if task_completed else 'No'}")
            # print(f"   Avg time (last 10): {avg_time:.2f}s")
            # print(f"   Total training time: {self.total_training_time/60:.1f}min")
            
            # Record task completion if successful
            if task_completed:
                self.task_completion_times.append(episode_duration)
                success_rate = len(self.task_completion_times) / self.episode_count * 100
                avg_completion_time = np.mean(self.task_completion_times)
                print(f"   Success rate: {success_rate:.1f}%")
                print(f"   Avg completion time: {avg_completion_time:.2f}s")
        
        # Start new episode timing
        self.episode_start_time = time.time()

        # Remove existing target bodies
        for body_id in getattr(self, 'target_body_ids', []):
            try:
                p.removeBody(body_id, physicsClientId=self.CLIENT)
            except:
                pass
        self.target_body_ids = []

        self.time_at_target = 0
        
        # Reset yaw target
        if hasattr(self, 'target_yaw'):
            delattr(self, 'target_yaw')
        
        # Randomize start and target positions if enabled
        if self.RANDOMIZE_INIT:
            self.START_POS = np.array([
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(0.8, 1.2)
            ])
            
            while True:
                self.TARGET_POS = np.array([
                    np.random.uniform(2.0, 4.0),
                    np.random.uniform(2.0, 4.0),
                    np.random.uniform(0.5, 2.0)
                ])
                if np.linalg.norm(self.TARGET_POS - self.START_POS) > 2.0:
                    break
        
        self.last_distance_to_target = np.linalg.norm(self.TARGET_POS - self.START_POS)
        self.initial_distance = self.last_distance_to_target
        if self.initial_distance < 1e-6:
            self.initial_distance = 1e-6

        super().reset(seed=seed, options=options)
        
        # Reset drone position and orientation
        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            self.START_POS,
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT
        )
        
        self._addObstacles()
        
        return self._computeObs(), self._computeInfo()

    def step(self, action):
        """Execute one environment step with optional noise injection."""
        # Apply action noise if enabled
        if self.enable_noise and self.noise_manager is not None:
            action = self.noise_manager.add_action_noise(action)
            self.noise_manager.step()
        
        self.total_steps += 1
        
        obs, reward, terminated, truncated, info = super().step(action)
        
        if self.GUI:
            keys = p.getKeyboardEvents(physicsClientId=self.CLIENT)
            # 'C': Toggle camera mode
            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                self.toggle_camera_mode()
            # '=': Increase camera distance
            elif ord('=') in keys and keys[ord('=')] & p.KEY_WAS_TRIGGERED:
                self.set_camera_distance(self.camera_distance + 0.1)
            # '-': Decrease camera distance
            elif ord('-') in keys and keys[ord('-')] & p.KEY_WAS_TRIGGERED:
                self.set_camera_distance(self.camera_distance - 0.1)
        
        # Update visualization
        if self.GUI and self.step_counter % 5 == 0:
            self._drawConnectionLine()
        if self.GUI and self.step_counter % 1 == 0:
            self._updateCamera()
        
        # Update noise UI if enabled
        if self.enable_noise and self.noise_manager is not None and self.GUI and self.show_noise_ui:
            if self.step_counter % 10 == 0:  # Update every 10 steps to avoid performance issues
                self.noise_manager._update_ui_display()
        
        return obs, reward, terminated, truncated, info

    # ======================== Gaussian Noise Control Methods ========================
    
    def set_training_mode(self, training: bool):
        """Set training/evaluation mode for noise management."""
        if self.noise_manager is not None:
            self.noise_manager.set_training_mode(training)
            print(f"Noise mode set to: {'Training' if training else 'Evaluation'}")
    
    def get_noise_statistics(self):
        """Get noise statistics if noise manager is available."""
        if self.noise_manager is not None:
            return self.noise_manager.get_noise_statistics()
        return {}
    
    def print_noise_stats(self):
        """Print noise statistics."""
        if self.noise_manager is not None:
            print("\nGaussian Noise Statistics:")
            self.noise_manager.print_noise_statistics()
        else:
            print("No noise statistics available (noise disabled)")
    
    def toggle_noise_ui(self):
        """Toggle noise UI display."""
        if self.enable_noise and self.noise_manager is not None:
            self.show_noise_ui = not self.show_noise_ui
            if self.show_noise_ui and self.GUI:
                self.noise_manager._setup_ui_display()
            print(f"Noise UI {'enabled' if self.show_noise_ui else 'disabled'}")
        else:
            print("Warning: Noise not enabled or not available")
    
    def get_noise_info(self):
        """Get noise configuration information."""
        if self.noise_manager is not None:
            return {
                'enabled': self.enable_noise,
                'level': self.noise_level,
                'decay_enabled': self.noise_decay,
                'ui_enabled': self.show_noise_ui,
                'training_progress': self.noise_manager.training_progress_callback() if self.noise_manager.training_progress_callback else 0,
                'total_episodes': self.total_episodes,
                'total_steps': self.total_steps
            }
        else:
            return {'enabled': False, 'level': 'none'}

    