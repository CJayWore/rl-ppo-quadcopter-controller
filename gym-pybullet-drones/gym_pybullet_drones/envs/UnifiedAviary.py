import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class UnifiedAviary(BaseRLAviary):
    """
    Unified RL environment: Navigate to target while avoiding obstacles and hover at destination.
    
    This environment combines three capabilities:
    1. Path planning: Navigate towards a target position
    2. Obstacle avoidance: Avoid obstacles along the path
    3. Hovering: Maintain stable hovering at the target position
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
                 target_radius: float = 0.15,  # Target area radius
                 hover_threshold: float = 0.05,  # Hover precision threshold
                 episode_len_sec: int = 30):
        
        # Task
        self.RANDOMIZE_INIT = randomize_init
        self.NUM_OBSTACLES = num_obstacles
        self.OBSTACLE_RADIUS = obstacle_radius
        self.SENSING_RANGE = sensing_range
        self.TARGET_RADIUS = target_radius
        self.HOVER_THRESHOLD = hover_threshold
        self.EPISODE_LEN_SEC = episode_len_sec
        
        # Task state
        self.TASK_STATE = "NAVIGATE"  # "NAVIGATE", "APPROACHING", "HOVERING", "COMPLETED"
        self.START_POS = np.array([0, 0, 1])
        self.TARGET_POS = np.array([3, 3, 1.5])
        self.obstacle_positions = []
        self.obstacle_ids = []
        
        # Performance metrics
        self.time_at_target = 0
        self.required_hover_time = 3.0  # required time to hover at target
        self.navigation_progress = 0.0
        self.last_distance_to_target = 0.0
        
        # Visualisation
        self.target_visual_id = None
        self.connection_line_id = None
        self.path_markers = []

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
        # # 扩展观察空间以包含环境感知信息
        # original_obs_dim = self.observation_space.shape[0]
        # num_lidar_rays = 12  # lidar rays in 12 directions
        # additional_info = 6    # relative target position(3) + task state information(3)
        # new_obs_dim = original_obs_dim + num_lidar_rays + additional_info
        
        # self.observation_space = spaces.Box(low=-np.inf, 
        #                                     high=np.inf, 
        #                                     shape=(new_obs_dim,), 
        #                                     dtype=np.float32)

    def _setupObservationSpace(self):
        """设置扩展的观察空间"""
        # 获取原始观察空间的维度
        original_obs_dim = self.observation_space.shape[0]
        
        # 添加的观察维度
        num_lidar_rays = 12     # 12个方向的激光雷达
        target_info = 3         # 相对目标位置(3)
        task_state_info = 3     # 任务状态编码(3)
        
        # 计算新的观察空间维度
        new_obs_dim = original_obs_dim + num_lidar_rays + target_info + task_state_info
        
        # 重新定义观察空间
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(new_obs_dim,), 
            dtype=np.float32
        )
        
        print(f"[UnifiedAviary] Original obs dim: {original_obs_dim}")
        print(f"[UnifiedAviary] New obs dim: {new_obs_dim}")
        print(f"[UnifiedAviary] Added: {num_lidar_rays} lidar + {target_info} target + {task_state_info} task state")

    def _addObstacles(self):
        """添加随机障碍物和目标可视化"""
        # clear existing obstacles
        for obs_id in self.obstacle_ids:
            p.removeBody(obs_id, physicsClientId=self.CLIENT)
        self.obstacle_ids = []
        self.obstacle_positions = []

        # Randomly place obstacles between start and end points
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
        """Visualize the target position and area"""
        if self.GUI:
            target_visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.1,
                rgbaColor=[0, 1, 0, 0.8],  # 绿色半透明
                physicsClientId=self.CLIENT
            )
            
            self.target_visual_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,  # No collision shape
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
                baseCollisionShapeIndex=-1,  # No collision shape
                baseVisualShapeIndex=target_area_visual,
                basePosition=self.TARGET_POS,
                physicsClientId=self.CLIENT
            )

            self.target_body_ids = [self.target_visual_id, self.target_area_id]

    def _get_lidar_readings(self, drone_pos):
        """获取12个方向的激光雷达读数"""
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
                direction = np.array([np.cos(angle), 0, np.sin(angle)]) # Forward tilt
            
            ray_start = drone_pos
            ray_end = drone_pos + direction * max_distance
            
            hit_info = p.rayTest(ray_start, ray_end, physicsClientId=self.CLIENT)
            
            if hit_info[0][0] != -1:
                hit_body_id = hit_info[0][0]
                
                # 过滤掉目标点的 body ID
                if hit_body_id in self.target_body_ids:
                    distances.append(max_distance)  # 忽略目标点，视为无障碍
                else:
                    hit_distance = hit_info[0][2] * max_distance
                    distances.append(hit_distance)
            else:
                distances.append(max_distance)
        
        return np.array(distances)

    def _update_task_state(self, drone_pos):
        distance_to_target = np.linalg.norm(self.TARGET_POS - drone_pos)
        
        if self.TASK_STATE == "NAVIGATE":
            if distance_to_target < self.TARGET_RADIUS:
                self.TASK_STATE = "APPROACHING"
                print(f"🎯 Entering target area! Distance: {distance_to_target:.3f}m")
        
        elif self.TASK_STATE == "APPROACHING":
            if distance_to_target < self.HOVER_THRESHOLD:
                self.TASK_STATE = "HOVERING"
                self.time_at_target = 0
                print(f"🎪 Started hovering! Distance: {distance_to_target:.3f}m")
            elif distance_to_target > self.TARGET_RADIUS:
                self.TASK_STATE = "NAVIGATE"
                print("⚠️ Left target area, resuming navigation...")
        
        elif self.TASK_STATE == "HOVERING":
            if distance_to_target < self.HOVER_THRESHOLD:
                self.time_at_target += 1.0 / self.CTRL_FREQ
                if self.time_at_target >= self.required_hover_time:
                    self.TASK_STATE = "COMPLETED"
                    print(f"✅ Task completed! Hovered for {self.time_at_target:.1f}s")
            else:
                self.TASK_STATE = "NAVIGATE"
                self.time_at_target = 0
                print("⚠️ Lost hover position, resuming navigation...")

    def _computeObs(self):
        # """计算增强的观察值"""
        # # 获取基础观察
        # base_obs = super()._computeObs()
        
        # # 确保base_obs是1维数组
        # if isinstance(base_obs, np.ndarray):
        #     if base_obs.ndim > 1:
        #         base_obs = base_obs.flatten()
        # else:
        #     base_obs = np.array(base_obs).flatten()
        
        # # 获取当前状态
        # state = self._getDroneStateVector(0)
        # current_pos = state[0:3]
        
        # # 获取激光雷达读数
        # lidar_readings = self._get_lidar_readings(current_pos)
        
        # # 计算相对目标位置
        # relative_target = self.TARGET_POS - current_pos
        
        # # 任务状态编码
        # task_state_encoding = {
        #     "NAVIGATE": [1, 0, 0],
        #     "APPROACHING": [0, 1, 0],
        #     "HOVERING": [0, 0, 1],
        #     "COMPLETED": [0, 0, 0]
        # }
        # task_info = np.array(task_state_encoding[self.TASK_STATE])
        
        # # 组合观察
        # enhanced_obs = np.concatenate([
        #     base_obs,
        #     lidar_readings,
        #     relative_target,
        #     task_info
        # ])
        
        # return enhanced_obs.astype(np.float32)
        return super()._computeObs()

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        current_vel = state[10:13]
        
        self._update_task_state(current_pos)
        total_reward = 0.0
        
        # 1. Navigation reward - based on distance to target
        current_distance = np.linalg.norm(self.TARGET_POS - current_pos)
        navigation_reward = (self.last_distance_to_target - current_distance) * 15.0
        self.last_distance_to_target = current_distance
        
        # 2. Obstacle avoidance penalty - based on distance to obstacles
        obstacle_penalty = 0
        min_obstacle_distance = float('inf')
        
        for obs_pos in self.obstacle_positions:
            dist_to_obstacle = np.linalg.norm(current_pos - obs_pos)
            min_obstacle_distance = min(min_obstacle_distance, dist_to_obstacle)
            
            safety_distance = self.OBSTACLE_RADIUS + 0.4
            danger_distance = self.OBSTACLE_RADIUS + 0.2

            # Danger zone penalty
            if dist_to_obstacle < safety_distance:
                safety_factor = max(0, (safety_distance - dist_to_obstacle) / (safety_distance - danger_distance))
                obstacle_penalty -= 5.0 * safety_factor

            # Crash penalty
            if dist_to_obstacle < self.OBSTACLE_RADIUS + 0.1:
                obstacle_penalty -= 100.0

        # 3. Task state rewards
        if self.TASK_STATE == "NAVIGATE":
            # Navigation: Encourage moving towards the target
            direction_to_target = self.TARGET_POS - current_pos
            if np.linalg.norm(direction_to_target) > 0:
                direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
                velocity_direction = current_vel / (np.linalg.norm(current_vel) + 1e-6)
                direction_reward = 2.0 * np.dot(direction_to_target, velocity_direction)
            else:
                direction_reward = 0
            
            distance_reward = 10.0 * np.exp(-current_distance * 1.5)
            attitude_stability = -2.0 * (abs(state[7]) + abs(state[8]))  # roll, pitch
            angular_stability = -1.0 * np.linalg.norm(state[13:16])     # angular velocity
            task_reward = direction_reward + distance_reward + attitude_stability + angular_stability
            
        elif self.TASK_STATE == "APPROACHING":
            # Approaching: Encourage approaching the target
            approach_reward = 10.0 * np.exp(-current_distance * 10.0)
            
            # Speed penalty: discourage high speeds
            speed_penalty = -2.0 * np.linalg.norm(current_vel)

            attitude_stability = -3.0 * (abs(state[7]) + abs(state[8]))
            angular_stability = -2.0 * np.linalg.norm(state[13:16])
            
            task_reward = approach_reward + speed_penalty + attitude_stability + angular_stability
            
        elif self.TASK_STATE == "HOVERING":
            # Hovering: Encourage stable hovering at the target position
            hover_reward = 30.0 * np.exp(-current_distance * 15.0)
            
            # Velocity and attitude penalties: discourage excessive movement
            velocity_penalty = -8.0 * np.linalg.norm(current_vel)
            attitude_penalty = -5.0 * (abs(state[7]) + abs(state[8]))  # roll, pitch
            angular_velocity_penalty = -3.0 * np.linalg.norm(state[13:16])
            
            # Hover time reward: encourage staying at the target
            hover_time_reward = self.time_at_target * 5.0
            
            task_reward = hover_reward + velocity_penalty + attitude_penalty + angular_velocity_penalty + hover_time_reward
            
        elif self.TASK_STATE == "COMPLETED":
            task_reward = 500.0
        
        else:
            task_reward = 0
        
        # 4. General penalties
        boundary_penalty = 0
        if (current_pos[0] < -2 or current_pos[0] > 5 or 
            current_pos[1] < -2 or current_pos[1] > 5 or
            current_pos[2] < 0.1 or current_pos[2] > 3.0):
            boundary_penalty = -100.0
        
        time_penalty = -0.05  # Encourage completing the task quickly
        
        total_reward = (
            navigation_reward +
            obstacle_penalty +
            task_reward +
            boundary_penalty +
            time_penalty
        )
        
        return total_reward

    def _computeTerminated(self):
        """Is the task completed?"""
        return self.TASK_STATE == "COMPLETED"

    def _computeTruncated(self):
        """Is the episode truncated?"""
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        
        # Hit obstacle detection
        for obs_pos in self.obstacle_positions:
            if np.linalg.norm(current_pos - obs_pos) < self.OBSTACLE_RADIUS + 0.05:
                return True
        
        # Border violation detection
        if (current_pos[0] < -2 or current_pos[0] > 5 or 
            current_pos[1] < -2 or current_pos[1] > 5 or
            current_pos[2] < 0.05 or current_pos[2] > 3.5):
            return True
        
        # Altitude violation detection
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
        
        return {
            "current_pos": current_pos,
            "target_pos": self.TARGET_POS,
            "distance_to_target": np.linalg.norm(current_pos - self.TARGET_POS),
            "task_state": self.TASK_STATE,
            "time_at_target": self.time_at_target,
            "navigation_progress": self.navigation_progress,
            "obstacle_positions": self.obstacle_positions
        }

    def reset(self, seed=None, options=None):
        '''reset state'''
        # Remove existing target bodies
        for body_id in getattr(self, 'target_body_ids', []):
            try:
                p.removeBody(body_id, physicsClientId=self.CLIENT)
            except:
                pass
        self.target_body_ids = []

        self.TASK_STATE = "NAVIGATE"
        self.time_at_target = 0
        # self.navigation_progress = 0.0
        
        # Randomise start and target positions if enabled
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
                    np.random.uniform(1.0, 2.0)
                ])
                if np.linalg.norm(self.TARGET_POS - self.START_POS) > 2.0:
                    break
        
        self.last_distance_to_target = np.linalg.norm(self.TARGET_POS - self.START_POS)
        
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
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update visualisation
        if self.GUI and self.step_counter % 10 == 0:
            self._drawConnectionLine()
        
        return obs, reward, terminated, truncated, info

    def _drawConnectionLine(self):
        if self.GUI:
            drone_state = self._getDroneStateVector(0)
            drone_pos = drone_state[0:3]
            
            if self.connection_line_id is not None:
                p.removeUserDebugItem(self.connection_line_id, physicsClientId=self.CLIENT)
            
            # 根据任务状态选择不同颜色
            if self.TASK_STATE == "NAVIGATE":
                color = [0, 0, 1]  # 蓝色
            elif self.TASK_STATE == "APPROACHING":
                color = [1, 1, 0]  # 黄色
            elif self.TASK_STATE == "HOVERING":
                color = [0, 1, 0]  # 绿色
            else:
                color = [1, 0, 1]  # 紫色
            
            self.connection_line_id = p.addUserDebugLine(
                lineFromXYZ=drone_pos,
                lineToXYZ=self.TARGET_POS,
                lineColorRGB=color,
                lineWidth=3,
                lifeTime=0.2,
                physicsClientId=self.CLIENT
            )