import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class ObstacleAviary(BaseRLAviary):
    """Single agent RL problem: navigate through obstacles."""

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
                 num_obstacles=5,
                 obstacle_radius=0.3,
                 sensing_range=2.0):
        
        self.NUM_OBSTACLES = num_obstacles
        self.OBSTACLE_RADIUS = obstacle_radius
        self.SENSING_RANGE = sensing_range
        self.obstacle_positions = []
        
        # 起点和终点
        self.START_POS = np.array([0, 0, 1])
        self.GOAL_POS = np.array([3, 3, 1])
        
        self.obstacle_ids = []

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
        original_obs_dim = self.observation_space.shape[1]
        num_lidar_rays = 8
        new_obs_dim = original_obs_dim + num_lidar_rays + 3
        
        self.observation_space = spaces.Box(low=-np.inf, 
                                            high=np.inf, 
                                            shape=(new_obs_dim,), 
                                            dtype=np.float32)


    def _addObstacles(self):
        """添加随机障碍物"""
        for obs_id in self.obstacle_ids:
            p.removeBody(obs_id, physicsClientId=self.CLIENT)
        self.obstacle_ids = []
        self.obstacle_positions = []

        # 在起点和终点之间的区域随机放置障碍物
        for i in range(self.NUM_OBSTACLES):
            # 随机生成障碍物位置（避免太接近起点和终点）
            while True:
                x = np.random.uniform(0.5, 2.5)
                y = np.random.uniform(0.5, 2.5)
                z = np.random.uniform(0.5, 1.5)
                pos = np.array([x, y, z])
                
                # 确保不要太接近起点和终点
                if (np.linalg.norm(pos - self.START_POS) > 0.5 and 
                    np.linalg.norm(pos - self.GOAL_POS) > 0.5):
                    break
            
            # 创建障碍物
            obstacle_id = p.createCollisionShape(p.GEOM_SPHERE, radius=self.OBSTACLE_RADIUS)
            visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=self.OBSTACLE_RADIUS,
                                          rgbaColor=[1, 0, 0, 0.8])
            
            body_id = p.createMultiBody(baseMass=0,
                                       baseCollisionShapeIndex=obstacle_id,
                                       baseVisualShapeIndex=visual_id,
                                       basePosition=pos,
                                       physicsClientId=self.CLIENT)
            
            self.obstacle_ids.append(body_id)
            self.obstacle_positions.append(pos)

    def _get_lidar_readings(self, drone_pos):
        """获取激光雷达读数（模拟）"""
        num_rays = 8  # 8个方向的射线
        max_distance = self.SENSING_RANGE
        
        distances = []
        for i in range(num_rays):
            angle = 2 * np.pi * i / num_rays
            direction = np.array([np.cos(angle), np.sin(angle), 0])
            
            ray_start = drone_pos
            ray_end = drone_pos + direction * max_distance
            
            # 使用 PyBullet 的射线检测
            hit_info = p.rayTest(ray_start, ray_end, physicsClientId=self.CLIENT)
            
            if hit_info[0][0] != -1:  # 如果射线击中了物体
                hit_distance = hit_info[0][2] * max_distance
                distances.append(hit_distance)
            else:
                distances.append(max_distance)
        
        return np.array(distances)

    def _computeObs(self):
        """计算观测值（包含激光雷达信息）"""
        # 获取基础观测
        obs = super()._computeObs()
        
        # 获取当前位置
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        
        # 获取激光雷达读数
        lidar_readings = self._get_lidar_readings(current_pos)
        
        # 目标位置相对坐标
        relative_goal = self.GOAL_POS - current_pos
        
        # 组合观测
        if isinstance(obs, np.ndarray):
            # 如果是一维数组，直接拼接
            if obs.ndim == 1:
                enhanced_obs = np.concatenate([obs, lidar_readings, relative_goal])
            else:
                # 如果是二维数组，取第一个无人机的观测
                enhanced_obs = np.concatenate([obs[0], lidar_readings, relative_goal])
        else:
            # 如果是字典格式，需要相应处理
            enhanced_obs = obs
        
        return enhanced_obs

    def _computeReward(self):
        """计算奖励函数"""
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        current_vel = state[10:13]


        # Calculate the difference in distance to goal (previous vs current)
        prev_distance = self.last_distance_to_goal
        current_distance = np.linalg.norm(self.GOAL_POS - current_pos)
        goal_reward = (prev_distance - current_distance) * 10.0 # if the drone gets closer to the goal, it receives a positive reward
        
        self.last_distance_to_goal = current_distance
        
        # 避障奖励
        obstacle_penalty = 0
        for obs_pos in self.obstacle_positions:
            dist_to_obstacle = np.linalg.norm(current_pos - obs_pos)
            if dist_to_obstacle < self.OBSTACLE_RADIUS + 0.2:  # 危险区域
                obstacle_penalty -= 5 * np.exp(-(dist_to_obstacle - self.OBSTACLE_RADIUS))
        
        # 速度奖励（鼓励朝目标方向移动）
        direction_to_goal = self.GOAL_POS - current_pos
        if np.linalg.norm(direction_to_goal) > 0:
            direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal)
            velocity_direction = current_vel / (np.linalg.norm(current_vel) + 1e-6)
            direction_reward = 0.5 * np.dot(direction_to_goal, velocity_direction)
        else:
            direction_reward = 0
        
        # 高度惩罚（保持合理高度）
        height_penalty = -0.1 * abs(current_pos[2] - 1.0)
        
        # 姿态惩罚
        rpy = state[7:10]
        attitude_penalty = -0.1 * (abs(rpy[0]) + abs(rpy[1]))
        
        total_reward = goal_reward + obstacle_penalty + direction_reward + height_penalty + attitude_penalty
        
        return total_reward

    def _computeTerminated(self):
        """计算任务是否完成"""
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        
        # 到达目标
        if np.linalg.norm(current_pos - self.GOAL_POS) < 0.2:
            return True
        
        return False

    def _computeTruncated(self):
        """计算是否需要截断episode"""
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        
        # 撞击障碍物
        for obs_pos in self.obstacle_positions:
            if np.linalg.norm(current_pos - obs_pos) < self.OBSTACLE_RADIUS + 0.1:
                return True
        
        # 撞地或飞太高
        if current_pos[2] < 0.1 or current_pos[2] > 3.0:
            return True
        
        # 飞出边界
        if (abs(current_pos[0]) > 5 or abs(current_pos[1]) > 5):
            return True
        
        return False

    def _computeInfo(self):
        """计算额外信息"""
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        
        return {
            "current_pos": current_pos,
            "goal_pos": self.GOAL_POS,
            "distance_to_goal": np.linalg.norm(current_pos - self.GOAL_POS),
            "obstacle_positions": self.obstacle_positions
        }

    def reset(self, seed=None, options=None):
        """重置环境"""
        # 清除旧的障碍物
        # self.obstacle_positions = []
        
        # 重置后添加新的障碍物
        super().reset(seed=seed, options=options)
        self._addObstacles()
        self.last_distance_to_goal = np.linalg.norm(self.GOAL_POS - self.START_POS)
        
        return self._computeObs(), self._computeInfo()