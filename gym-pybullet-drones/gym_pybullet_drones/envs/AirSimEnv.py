import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import time
import math

class AirSimEnv(gym.Env):
    """
    AirSim Wrapper compatible with Stable-Baselines3 and the DRLAviary observation/action space.
    """
    def __init__(self, target_pos=np.array([3, 3, 1.5])):
        super(AirSimEnv, self).__init__()
        
        # 连接 AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        self.target_pos = target_pos
        
        # 强制与原有的 Unified 任务保持绝对一致
        # 原版模型之所以是 97，是因为 BaseRLAviary._observationSpace 如果是 72，再加上额外扩展的 25 就是 97
        self.obs_dim = 97
        self.original_obs_dim = self.obs_dim - 25 # (12 + 3 + 3 + 3 + 1 + 3)
        
        # 【动作空间】匹配 ActionType.RPM (4个电机的归一化输入 [-1, 1])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 【观测空间】
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        time.sleep(0.1) # 等待重置稳定
        
        obs = self._get_obs()
        info = {'target_pos': self.target_pos.tolist()}
        return obs, info

    def step(self, action):
        pwm_cmds = np.clip((action + 1.0) / 2.0, 0.0, 1.0)
        
        self.client.moveByMotorPWMsAsync(
            float(pwm_cmds[0]), 
            float(pwm_cmds[1]), 
            float(pwm_cmds[2]), 
            float(pwm_cmds[3]), 
            duration=0.03 # 匹配你的控制频率
        ).join()

        obs = self._get_obs()
        
        dist_to_target = np.linalg.norm(self._get_drone_pos() - self.target_pos)
        reward = -dist_to_target
        
        collision_info = self.client.simGetCollisionInfo()
        terminated = collision_info.has_collided
        truncated = False
        info = {'target_pos': self.target_pos.tolist()}
        
        return obs, reward, terminated, truncated, info

    def _get_drone_pos(self):
        kinematics = self.client.simGetGroundTruthKinematics()
        return np.array([kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val])

    def _get_obs(self):
        kinematics = self.client.simGetGroundTruthKinematics()
        
        pos = np.array([kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val])
        vel = np.array([kinematics.linear_velocity.x_val, kinematics.linear_velocity.y_val, kinematics.linear_velocity.z_val])
        ang_vel = np.array([kinematics.angular_velocity.x_val, kinematics.angular_velocity.y_val, kinematics.angular_velocity.z_val])
        
        pitch, roll, yaw = airsim.to_eularian_angles(kinematics.orientation)
        
        # 原生 DRLAviary 在 BaseRLAviary 中取 _getDroneStateVector() 返回长度为 20。然后经过 Kinematic 扩展组合变成了 72 长。
        # 这里为了快速运行旧网络，我们填入核心的 pos, euler, vel, ang_vel 数据，并把其余位用 0 填充以满足 72 维 (97 - 25 = 72)。
        core_obs = np.concatenate([pos, np.array([roll, pitch, yaw]), vel, ang_vel])
        pad_size = self.original_obs_dim - len(core_obs)
        pad = np.zeros(pad_size, dtype=np.float32)
        base_obs = np.concatenate([core_obs, pad])
        
        # ----拼凑扩展数据 (25维)----
        lidar_readings = np.full(12, 2.0)
        target_info_world = self.target_pos - pos
        
        cy = math.cos(yaw); sy = math.sin(yaw)
        R_yaw = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]]) 
        target_info_body = R_yaw.dot(target_info_world)
        velocity_body = R_yaw.dot(vel)
        
        dist = np.array([np.linalg.norm(target_info_world)])
        angle_to_target = math.atan2(target_info_world[1], target_info_world[0]) - yaw
        angle_info = np.array([angle_to_target, math.sin(angle_to_target), math.cos(angle_to_target)])
        
        full_obs = np.concatenate([
            base_obs, lidar_readings, target_info_world, 
            target_info_body, velocity_body, dist, angle_info
        ]).astype(np.float32)
        
        return full_obs