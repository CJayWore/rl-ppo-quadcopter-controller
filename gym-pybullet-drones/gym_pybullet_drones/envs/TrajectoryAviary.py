import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class TrajectoryAviary(BaseRLAviary):
    """Single agent RL problem: follow a trajectory path."""

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
                 trajectory_type="circle",
                 trajectory_radius=0.5,
                 trajectory_height=1.0,
                 max_distance_from_path=2.0):
        
        # 轨迹参数
        self.TRAJECTORY_TYPE = trajectory_type
        self.TRAJECTORY_RADIUS = trajectory_radius
        self.TRAJECTORY_HEIGHT = trajectory_height
        self.MAX_DISTANCE_FROM_PATH = max_distance_from_path
        
        # 时间步计数器
        self.step_counter = 0
        
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)
        
        original_obs_dim = self.observation_space.shape[0]
        new_obs_dim = original_obs_dim # + 3 + 3 # 如果添加了额外观测，取消这部分注释
        
        self.observation_space = spaces.Box(low=-np.inf, 
                                            high=np.inf, 
                                            shape=(new_obs_dim,), 
                                            dtype=np.float32)
        
        # 生成轨迹
        self._generate_trajectory()

    def _generate_trajectory(self):
        """生成参考轨迹"""
        total_steps = int(self.EPISODE_LEN_SEC * self.CTRL_FREQ)
        self.trajectory = np.zeros((total_steps, 3))
        
        if self.TRAJECTORY_TYPE == "circle":
            for i in range(total_steps):
                angle = 2 * np.pi * i / total_steps
                self.trajectory[i] = [
                    self.TRAJECTORY_RADIUS * np.cos(angle),
                    self.TRAJECTORY_RADIUS * np.sin(angle),
                    self.TRAJECTORY_HEIGHT
                ]
        elif self.TRAJECTORY_TYPE == "figure8":
            for i in range(total_steps):
                t = 2 * np.pi * i / total_steps
                self.trajectory[i] = [
                    self.TRAJECTORY_RADIUS * np.sin(t),
                    self.TRAJECTORY_RADIUS * np.sin(t) * np.cos(t),
                    self.TRAJECTORY_HEIGHT
                ]
        elif self.TRAJECTORY_TYPE == "waypoints":
            # 定义几个关键点
            waypoints = np.array([
                [0, 0, 1.0],
                [1, 0, 1.5],
                [1, 1, 1.0],
                [0, 1, 1.5],
                [-1, 0, 1.0],
                [0, -1, 1.5],
                [0, 0, 1.0]
            ])
            
            # 在关键点之间插值
            for i in range(total_steps):
                progress = i / total_steps * (len(waypoints) - 1)
                idx = int(progress)
                alpha = progress - idx
                
                if idx >= len(waypoints) - 1:
                    self.trajectory[i] = waypoints[-1]
                else:
                    self.trajectory[i] = (1 - alpha) * waypoints[idx] + alpha * waypoints[idx + 1]

    def _get_current_target(self):
        """获取当前目标位置"""
        if self.step_counter >= len(self.trajectory):
            return self.trajectory[-1]
        return self.trajectory[self.step_counter]

    def _computeReward(self):
        """计算奖励函数"""
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        current_vel = state[10:13]
        
        # 当前目标位置
        target_pos = self._get_current_target()
        
        # 位置误差奖励
        pos_error = np.linalg.norm(target_pos - current_pos)
        pos_reward = np.exp(-pos_error * 5)  # 距离越近奖励越高
        
        # 速度惩罚（鼓励平稳飞行）
        vel_penalty = -0.1 * np.linalg.norm(current_vel)
        
        # 轨迹跟踪奖励
        if self.step_counter < len(self.trajectory) - 1:
            next_target = self.trajectory[self.step_counter + 1]
            desired_direction = next_target - target_pos
            if np.linalg.norm(desired_direction) > 0:
                desired_direction = desired_direction / np.linalg.norm(desired_direction)
                actual_direction = current_vel / (np.linalg.norm(current_vel) + 1e-6)
                direction_reward = 0.5 * np.dot(desired_direction, actual_direction)
            else:
                direction_reward = 0
        else:
            direction_reward = 0
        
        # 姿态惩罚
        rpy = state[7:10]
        attitude_penalty = -0.1 * (abs(rpy[0]) + abs(rpy[1]))  # 限制roll和pitch
        
        # 高度维持奖励
        height_error = abs(current_pos[2] - target_pos[2])
        height_reward = np.exp(-height_error * 10)
        
        total_reward = pos_reward + vel_penalty + direction_reward + attitude_penalty + height_reward
        
        return total_reward

    def _computeTerminated(self):
        """计算任务是否完成"""
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        
        # 如果到达轨迹末端附近，认为任务完成
        if self.step_counter >= len(self.trajectory) - 10:
            final_pos = self.trajectory[-1]
            if np.linalg.norm(current_pos - final_pos) < 0.1:
                return True
        
        return False

    def _computeTruncated(self):
        """计算是否需要截断episode"""
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        
        # 检查是否偏离轨迹太远
        target_pos = self._get_current_target()
        if np.linalg.norm(current_pos - target_pos) > self.MAX_DISTANCE_FROM_PATH:
            return True
        
        # 检查是否撞地或飞太高
        if current_pos[2] < 0.1 or current_pos[2] > 3.0:
            return True
        
        # 检查姿态是否过大
        rpy = state[7:10]
        if abs(rpy[0]) > 0.5 or abs(rpy[1]) > 0.5:  # 30度
            return True
        
        return False

    def _computeInfo(self):
        """计算额外信息"""
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        target_pos = self._get_current_target()
        
        return {
            "current_pos": current_pos,
            "target_pos": target_pos,
            "distance_to_target": np.linalg.norm(current_pos - target_pos),
            "trajectory_progress": self.step_counter / len(self.trajectory)
        }

    def reset(self, seed=None, options=None):
        """重置环境"""
        self.step_counter = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        """执行一步"""
        obs, reward, terminated, truncated, info = super().step(action)
        self.step_counter += 1
        return obs, reward, terminated, truncated, info

    def _addObstacles(self):
        """添加障碍物（可选）"""
        if self.TRAJECTORY_TYPE == "waypoints":
            # 添加一些障碍物
            obstacles = [
                [0.5, 0.5, 0.5],  # 位置
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5]
            ]
            
            for i, pos in enumerate(obstacles):
                p.loadURDF("cube_small.urdf", pos, 
                          p.getQuaternionFromEuler([0, 0, 0]),
                          physicsClientId=self.CLIENT)