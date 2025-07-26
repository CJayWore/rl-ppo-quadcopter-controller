import dis
import numpy as np
import pybullet as p
import time
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
                 hover_threshold: float = 0.5,  # Hover precision threshold
                 episode_len_sec: int = 30,
                 enable_obstacles: bool = True):
        
        # Task
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
        if self.ENABLE_OBSTACLES:
            print("Adding obstacles...")
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
        else:
            print(f"⭕ No obstacles added - obstacle-free training mode")

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

    def _update_hover_timer(self, drone_pos):
        """更新悬停计时器（不影响奖励状态）"""
        distance_to_target = np.linalg.norm(self.TARGET_POS - drone_pos)
        
        if distance_to_target < self.HOVER_THRESHOLD:
            self.time_at_target += 1.0 / self.CTRL_FREQ
        else:
            self.time_at_target = 0  # 离开目标区域时重置

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

    '''

    TODO:
        - 添加高斯noise扰动

    '''
    def _navigationReward(self):
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        position_error = self.TARGET_POS - current_pos
        current_distance = np.linalg.norm(position_error)

        distance_reward = 0
        if current_distance >0.1:
            distance_reward = 10/(1e-6 + current_distance)
        else:
            distance_reward = 50 # Arrive at target with high reward

        # if current_distance >= 4.0:
        #     distance_reward = 100/(current_distance)  # 基于距离的奖励，距离越近奖励越高
        # else:
        #     distance_reward = np.exp(-distance_reward) * 100  # 距离小于4时，使用指数衰减奖励
        # distance_reward = -10 * (1 - np.exp(-0.1 * current_distance**2))
        delta_distance = (self.last_distance_to_target - current_distance)
        approaching_reward = 0
        if delta_distance > 0:
            approaching_reward = delta_distance * 10.0
        self.last_distance_to_target = current_distance

        # postion_precision_reward = 0
        k = 1.0
        x_penalty = 1.0-np.tanh(k * abs(position_error[0]))
        y_penalty = 1.0-np.tanh(k * abs(position_error[1]))
        z_penalty = 1.0-np.tanh(k * abs(position_error[2]))
        
        # 给Z轴稍高权重但不过分
        position_precision_reward = (x_penalty + y_penalty + 1.5 * z_penalty)

        height_safety_reward = 0
        if current_pos[2] < 0.3:  # 危险低高度
            height_safety_reward = -5.0 * (0.3 - current_pos[2])
        elif current_pos[2] > 0.5:  # 安全高度
            height_safety_reward = 2.0

        navigation_reward = (
            distance_reward
            + approaching_reward
            + position_precision_reward
            + height_safety_reward
            )
        
        return navigation_reward, {
            'distance_reward': distance_reward,
            'approaching_reward': approaching_reward,
            'position_precision_reward': position_precision_reward,
            'height_safety_reward': height_safety_reward,
            'x_penalty': x_penalty,
            'y_penalty': y_penalty,
            'z_penalty': z_penalty
        }
    
    def _stabilityPenalty(self):
        state = self._getDroneStateVector(0)
        # current_pos = state[0:3]
        quaternion = state[3:7]  # 四元数表示的姿态
        rpy = state[7:10]
        current_vel = state[10:13]
        angular_vel = state[13:16]  # angular velocity
        last_action = state[16:20]  # last action (RPMs)

        # 姿态稳定性
        max_tilt = np.pi/12  # 15度作为参考
        roll_stability = max(0, 10.0 * (1.0 - abs(rpy[0]) / max_tilt))
        pitch_stability = max(0, 10.0 * (1.0 - abs(rpy[1]) / max_tilt))
        attitude_penalty = roll_stability + pitch_stability

        max_ang_vel = 2.0 

        angular_velocity_reward = 5.0 * max(0, 1.0 - np.linalg.norm(angular_vel) / max_ang_vel)
        # linear_velocity_penalty = -0.5 * (current_vel[0]**2 + current_vel[1]**2 + current_vel[2]**2)
        
        # RPM smoothness penalty using available last_action
        rpm_smoothness_penalty = -0.1 * np.var(last_action) / (np.mean(last_action) + 1e-6)  # normalized by mean RPM

        stability_penalty = (
            attitude_penalty +
            angular_velocity_reward 
            # rpm_smoothness_penalty
        )

        return stability_penalty, {
            'attitude_penalty': attitude_penalty,
            'angular_velocity_reward': angular_velocity_reward,
            'rpm_smoothness_penalty': rpm_smoothness_penalty,
            'rpy': rpy,
            'angular_vel_norm': np.linalg.norm(angular_vel)
        }

    def _hoveringReward(self):
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        current_vel = state[10:13]
        
        distance_to_target = np.linalg.norm(current_pos - self.TARGET_POS)
        velocity_norm = np.linalg.norm(current_vel)
        # 悬停稳定性奖励 - 接近目标时奖励低速度
        # speed_reward = 0
        # if distance_to_target < self.HOVER_THRESHOLD:
        #     speed_reward = 100.0/(1e-6 + velocity_norm)  # 接近目标时速度越低奖励越高
        
        ################################################################################################
        MAX_HOVER_REWARD = 10.0  # Max reward for a perfect hover at the target center.
        
        # We set the distance decay so the reward is half its max at the HOVER_THRESHOLD boundary.
        DISTANCE_DECAY = np.log(2) / (self.HOVER_THRESHOLD**2)
        
        # This controls how strongly velocity is penalized. Higher value = more penalty for speed.
        VELOCITY_DECAY = 3.0

        # Calculate a distance-based factor (0 to 1) using a Gaussian function.
        distance_factor = np.exp(-DISTANCE_DECAY * distance_to_target**2)
        
        # Calculate a velocity-based factor (0 to 1) using another Gaussian function.
        velocity_factor = np.exp(-VELOCITY_DECAY * velocity_norm**2)
        
        # The final reward is the product of these smooth factors.
        speed_reward = MAX_HOVER_REWARD * distance_factor * velocity_factor
        ################################################################################################

        # 悬停时间奖励 - 在目标附近停留的时间
        hover_time_reward = 0
        if distance_to_target < self.HOVER_THRESHOLD:
            # 使用 min() 函数来给奖励设置一个上限，例如20.0
            # 这样既能鼓励持续悬停，又不会让奖励无限增长
            hover_time_reward = min(20.0, self.time_at_target * 2.0)
        
        hovering_reward = speed_reward + hover_time_reward

        return hovering_reward, {
            'speed_reward': speed_reward,
            'hover_time_reward': hover_time_reward,
            'velocity_norm': velocity_norm,
            'distance_to_target': distance_to_target,
            'in_hover_zone': distance_to_target < self.HOVER_THRESHOLD
        }

    def _obstacleAvoidanceReward(self):
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        
        obstacle_penalty = 0
        if self.ENABLE_OBSTACLES and len(self.obstacle_positions) > 0:
            critical_distance = self.OBSTACLE_RADIUS + 0.3
            for obs_pos in self.obstacle_positions:
                dist_to_obstacle = np.linalg.norm(current_pos - obs_pos)
                if dist_to_obstacle < critical_distance:
                    # 使用指数函数创建强烈的避障信号
                    penalty_factor = np.exp(-(dist_to_obstacle - self.OBSTACLE_RADIUS) * 5.0)
                    obstacle_penalty -= 10.0 * penalty_factor
                    
                    # 如果非常接近障碍物，给予额外的强烈惩罚
                    if dist_to_obstacle < self.OBSTACLE_RADIUS + 0.05:
                        obstacle_penalty -= 100.0
        
        return obstacle_penalty
    

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        current_vel = state[10:13]
        current_distance = np.linalg.norm(self.TARGET_POS - current_pos)

        # 计算各个奖励组件
        navigation_reward, nav_details = self._navigationReward()
        stability_reward, stab_details = self._stabilityPenalty()
        hovering_reward, hover_details = self._hoveringReward()
        obstacle_reward = self._obstacleAvoidanceReward()
        
        
        # 任务完成奖励
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

        # 详细调试输出
        # if self.step_counter % 100 == 0:
        #     obstacle_info = f", Obstacles: {len(self.obstacle_positions)}" if self.ENABLE_OBSTACLES else ", No obstacles"
        #     print(f"\n--------------------Environments-------------------- "
        #           f"\nTarget: {self.TARGET_POS}, \nDrone Pos: {current_pos}, "
        #           f"\nDistance: {current_distance:.3f},"
        #           f"Speed: {np.linalg.norm(current_vel):.2f}, "
        #           f"Hover time: {self.time_at_target:.1f}s{obstacle_info}")
            
        #     print(f"\n--------------------Rewards--------------------")
        #     print(f"Navigation ({navigation_reward:.2f}):")
        #     print(f"  - Distance: {nav_details['distance_reward']:.2f}")
        #     print(f"  - Approaching: {nav_details['approaching_reward']:.2f}")
        #     print(f"  - Position Precision: {nav_details['position_precision_reward']:.2f}")
        #     print(f"    * X penalty: {nav_details['x_penalty']:.3f}")
        #     print(f"    * Y penalty: {nav_details['y_penalty']:.3f}")
        #     print(f"    * Z penalty: {nav_details['z_penalty']:.3f}")
            
        #     print(f"Stability ({stability_reward:.2f}):")
        #     print(f"  - Attitude: {stab_details['attitude_penalty']:.2f} (RPY: {stab_details['rpy']})")
        #     print(f"  - Angular Vel: {stab_details['angular_velocity_reward']:.2f} (|ω|: {stab_details['angular_vel_norm']:.3f})")
        #     print(f"  - RPM Smoothness: {stab_details['rpm_smoothness_penalty']:.2f}")

        #     print(f"Hovering ({hovering_reward:.2f}):")
        #     print(f"  - Speed: {hover_details['speed_reward']:.2f} (|v|: {hover_details['velocity_norm']:.3f})")
        #     print(f"  - Hover Time: {hover_details['hover_time_reward']:.2f} (in zone: {hover_details['in_hover_zone']})")
            
        #     print(f"Obstacles ({obstacle_reward:.2f}):")

            
        #     print(f"Completion: {completion_reward:.2f}")
        #     print(f"TOTAL REWARD: {total_reward:.2f}")
        
        return total_reward

    def _computeTerminated(self):
        """Is the task completed?"""
        return self.time_at_target >= self.required_hover_time

    def _computeTruncated(self):
        """Is the episode truncated?"""
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        
        # Hit obstacle detection
        if self.ENABLE_OBSTACLES:
            for obs_pos in self.obstacle_positions:
                if np.linalg.norm(current_pos - obs_pos) < self.OBSTACLE_RADIUS + 0.05:
                    print(f"🚫 Truncated because of Obstacle Hit")
                    return True
        
        # Border violation detection
        if (current_pos[0] < -2 or current_pos[0] > 5 or 
            current_pos[1] < -2 or current_pos[1] > 5 or
            current_pos[2] < 0.05 or current_pos[2] > 3.5):
            print(f"🚫 Truncated because of Border Violation")
            return True
        
        # Altitude violation detection
        rpy = state[7:10]
        if abs(rpy[0]) > 1.0 or abs(rpy[1]) > 1.0:  # 57 degrees
            print(f"🚫 Truncated because of Altitude Violation")
            return True
        
        # Time limit detection
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            print(f"🚫 Truncated because of Time Limit")
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
        '''reset state'''
        # 记录上一个episode的结束时间
        if self.episode_start_time is not None:
            episode_duration = time.time() - self.episode_start_time
            self.episode_times.append(episode_duration)
            self.total_training_time += episode_duration
            
            # 打印episode统计信息
            self.episode_count += 1
            avg_time = np.mean(self.episode_times[-10:]) if len(self.episode_times) >= 10 else np.mean(self.episode_times)
            
            task_completed = self.time_at_target >= self.required_hover_time

            print(f"📊 Episode {self.episode_count} completed:")
            print(f"   Duration: {episode_duration:.2f}s")
            print(f"   Hover time: {self.time_at_target:.1f}s")
            print(f"   Task completed: {'✅ Yes' if task_completed else '❌ No'}")
            print(f"   Avg time (last 10): {avg_time:.2f}s")
            print(f"   Total training time: {self.total_training_time/60:.1f}min")
            
            # 如果任务完成，记录完成时间
            if task_completed:
                self.task_completion_times.append(episode_duration)
                success_rate = len(self.task_completion_times) / self.episode_count * 100
                avg_completion_time = np.mean(self.task_completion_times)
                print(f"   Success rate: {success_rate:.1f}%")
                print(f"   Avg completion time: {avg_completion_time:.2f}s")
        
        # 开始新episode的计时
        self.episode_start_time = time.time()

        # Remove existing target bodies
        for body_id in getattr(self, 'target_body_ids', []):
            try:
                p.removeBody(body_id, physicsClientId=self.CLIENT)
            except:
                pass
        self.target_body_ids = []

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
            if self.time_at_target >= self.required_hover_time:
                color = [0, 1, 0]  # 绿色 - 任务完成
            elif self.time_at_target > 0:
                color = [1, 1, 0]  # 黄色 - 正在悬停
            else:
                color = [0, 0, 1]  # 蓝色 - 导航中
            
            self.connection_line_id = p.addUserDebugLine(
                lineFromXYZ=drone_pos,
                lineToXYZ=self.TARGET_POS,
                lineColorRGB=color,
                lineWidth=3,
                lifeTime=0.2,
                physicsClientId=self.CLIENT
            )