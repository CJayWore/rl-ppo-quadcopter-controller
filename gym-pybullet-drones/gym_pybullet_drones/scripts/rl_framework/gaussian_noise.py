"""
Gaussian Noise Module for Drone RL Training

This module provides various types of Gaussian noise to simulate real-world disturbances
during training, making the agent more robust to sensor noise, wind, and other uncertainties.

"""

import numpy as np
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum


class NoiseType(Enum):
    """Types of noise that can be applied"""
    SENSOR_NOISE = "sensor"           # 传感器噪声
    WIND_DISTURBANCE = "wind"         # 风扰动
    MOTOR_NOISE = "motor"             # 电机噪声  
    OBSERVATION_NOISE = "observation" # 观测噪声
    ACTION_NOISE = "action"           # 动作噪声
    POSITION_DRIFT = "position"       # 位置漂移
    ATTITUDE_NOISE = "attitude"       # 姿态噪声


@dataclass
class NoiseConfig:
    """Configuration for different types of noise"""
    enabled: bool = True
    std_dev: float = 0.01              # 标准差
    mean: float = 0.0                  # 均值
    scale_factor: float = 1.0          # 缩放因子
    decay_rate: float = 1.0            # 衰减率（随训练进度降低噪声）
    min_std_dev: float = 0.001         # 最小标准差
    adaptive: bool = False             # 是否自适应调整
    clip_range: Optional[Tuple[float, float]] = None  # 裁剪范围


class GaussianNoiseManager:
    """
    Gaussian Noise Manager for Drone RL Training
    
    This class manages various types of Gaussian noise to simulate real-world
    disturbances and improve the robustness of the trained agent.
    """
    
    def __init__(self, 
                 training_progress_callback: Optional[callable] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize the Gaussian Noise Manager
        
        Args:
            training_progress_callback: Function that returns training progress [0,1]
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.training_progress_callback = training_progress_callback
        self.step_count = 0
        self.episode_count = 0
        
        # 预定义的噪声配置
        self.noise_configs = self._setup_default_noise_configs()
        
        # 噪声历史记录（用于分析和调试）
        self.noise_history = {noise_type.value: [] for noise_type in NoiseType}
        
        print("🔊 Gaussian Noise Manager initialized")
        self._print_noise_summary()

    def _setup_default_noise_configs(self) -> Dict[str, NoiseConfig]:
        """Setup default noise configurations for different disturbance types"""
        return {
            # 传感器噪声：影响位置、速度、角速度测量
            NoiseType.SENSOR_NOISE.value: NoiseConfig(
                enabled=True,
                std_dev=0.02,           # 2cm位置误差，0.02m/s速度误差
                mean=0.0,
                scale_factor=1.0,
                decay_rate=0.8,         # 随训练进度减少
                min_std_dev=0.005,
                adaptive=True,
                clip_range=(-0.1, 0.1)
            ),
            
            # 风扰动：模拟环境中的风力影响
            NoiseType.WIND_DISTURBANCE.value: NoiseConfig(
                enabled=True,
                std_dev=0.5,            # 0.5m/s^2的加速度扰动
                mean=0.0,
                scale_factor=1.0,
                decay_rate=0.9,
                min_std_dev=0.1,
                adaptive=True,
                clip_range=(-2.0, 2.0)
            ),
            
            # 电机噪声：模拟电机响应的不确定性
            NoiseType.MOTOR_NOISE.value: NoiseConfig(
                enabled=True,
                std_dev=0.05,           # 5%的RPM噪声
                mean=0.0,
                scale_factor=1.0,
                decay_rate=0.95,
                min_std_dev=0.01,
                adaptive=False,
                clip_range=(-0.2, 0.2)
            ),
            
            # 观测噪声：影响整个观测向量
            NoiseType.OBSERVATION_NOISE.value: NoiseConfig(
                enabled=True,
                std_dev=0.01,           # 小幅观测噪声
                mean=0.0,
                scale_factor=1.0,
                decay_rate=0.85,
                min_std_dev=0.002,
                adaptive=True,
                clip_range=(-0.05, 0.05)
            ),
            
            # 动作噪声：在动作执行时添加噪声
            NoiseType.ACTION_NOISE.value: NoiseConfig(
                enabled=True,
                std_dev=0.02,           # 2%的动作噪声
                mean=0.0,
                scale_factor=1.0,
                decay_rate=0.9,
                min_std_dev=0.005,
                adaptive=True,
                clip_range=(-0.1, 0.1)
            ),
            
            # 位置漂移：模拟GPS或定位系统的漂移
            NoiseType.POSITION_DRIFT.value: NoiseConfig(
                enabled=False,          # 默认关闭，较强的扰动
                std_dev=0.1,
                mean=0.0,
                scale_factor=1.0,
                decay_rate=0.7,
                min_std_dev=0.02,
                adaptive=True,
                clip_range=(-0.5, 0.5)
            ),
            
            # 姿态噪声：影响roll, pitch, yaw测量
            NoiseType.ATTITUDE_NOISE.value: NoiseConfig(
                enabled=True,
                std_dev=0.05,           # ~3度的角度噪声
                mean=0.0,
                scale_factor=1.0,
                decay_rate=0.8,
                min_std_dev=0.01,
                adaptive=True,
                clip_range=(-0.2, 0.2)  # ~11度限制
            )
        }

    def _print_noise_summary(self):
        """Print a summary of current noise configurations"""
        print("\n📊 Noise Configuration Summary:")
        print("-" * 60)
        for noise_type, config in self.noise_configs.items():
            status = "✅" if config.enabled else "❌"
            print(f"{status} {noise_type.ljust(15)}: σ={config.std_dev:.3f}, "
                  f"decay={config.decay_rate:.2f}, adaptive={config.adaptive}")
        print("-" * 60)

    def update_noise_config(self, noise_type: Union[str, NoiseType], **kwargs):
        """
        Update noise configuration for a specific noise type
        
        Args:
            noise_type: Type of noise to update
            **kwargs: Configuration parameters to update
        """
        if isinstance(noise_type, NoiseType):
            noise_type = noise_type.value
            
        if noise_type not in self.noise_configs:
            raise ValueError(f"Unknown noise type: {noise_type}")
            
        config = self.noise_configs[noise_type]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"⚠️ Unknown config parameter: {key}")
                
        print(f"🔧 Updated {noise_type} noise config: {kwargs}")

    def get_current_std_dev(self, noise_type: str) -> float:
        """
        Get current standard deviation considering decay and training progress
        
        Args:
            noise_type: Type of noise
            
        Returns:
            Current effective standard deviation
        """
        config = self.noise_configs[noise_type]
        
        if not config.enabled:
            return 0.0
            
        # 基础标准差
        base_std = config.std_dev
        
        # 如果启用衰减和自适应调整
        if config.adaptive and self.training_progress_callback:
            progress = self.training_progress_callback()
            # progress从0到1，噪声从full到min_std_dev
            decay_factor = config.decay_rate ** progress
            current_std = max(config.min_std_dev, base_std * decay_factor)
        else:
            current_std = base_std
            
        return current_std

    def add_sensor_noise(self, 
                        position: np.ndarray, 
                        velocity: np.ndarray, 
                        angular_velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Add sensor noise to position, velocity, and angular velocity measurements
        
        Args:
            position: Position vector [x, y, z]
            velocity: Velocity vector [vx, vy, vz]  
            angular_velocity: Angular velocity [wx, wy, wz]
            
        Returns:
            Noisy position, velocity, and angular velocity
        """
        std_dev = self.get_current_std_dev(NoiseType.SENSOR_NOISE.value)
        config = self.noise_configs[NoiseType.SENSOR_NOISE.value]
        
        if std_dev == 0.0:
            return position, velocity, angular_velocity
            
        # 生成噪声
        pos_noise = np.random.normal(config.mean, std_dev * 0.5, 3)  # 位置噪声小一些
        vel_noise = np.random.normal(config.mean, std_dev, 3)
        ang_vel_noise = np.random.normal(config.mean, std_dev * 2.0, 3)  # 角速度噪声大一些
        
        # 应用裁剪
        if config.clip_range:
            pos_noise = np.clip(pos_noise, *config.clip_range)
            vel_noise = np.clip(vel_noise, *config.clip_range)  
            ang_vel_noise = np.clip(ang_vel_noise, *config.clip_range)
        
        # 记录噪声历史
        self.noise_history[NoiseType.SENSOR_NOISE.value].append({
            'step': self.step_count,
            'pos_noise_norm': np.linalg.norm(pos_noise),
            'vel_noise_norm': np.linalg.norm(vel_noise),
            'ang_vel_noise_norm': np.linalg.norm(ang_vel_noise)
        })
        
        return position + pos_noise, velocity + vel_noise, angular_velocity + ang_vel_noise

    def add_wind_disturbance(self, current_velocity: np.ndarray) -> np.ndarray:
        """
        Add wind disturbance to simulate environmental wind effects
        
        Args:
            current_velocity: Current velocity of the drone
            
        Returns:
            Wind disturbance force/acceleration vector
        """
        std_dev = self.get_current_std_dev(NoiseType.WIND_DISTURBANCE.value)
        config = self.noise_configs[NoiseType.WIND_DISTURBANCE.value]
        
        if std_dev == 0.0:
            return np.zeros(3)
            
        # 生成随机风向和强度
        wind_disturbance = np.random.normal(config.mean, std_dev, 3)
        
        # 模拟更真实的风：水平方向强一些，垂直方向弱一些
        wind_disturbance[2] *= 0.3  # 垂直风力较小
        
        # 应用裁剪
        if config.clip_range:
            wind_disturbance = np.clip(wind_disturbance, *config.clip_range)
        
        # 记录噪声历史
        self.noise_history[NoiseType.WIND_DISTURBANCE.value].append({
            'step': self.step_count,
            'wind_force_norm': np.linalg.norm(wind_disturbance)
        })
        
        return wind_disturbance

    def add_motor_noise(self, motor_commands: np.ndarray) -> np.ndarray:
        """
        Add noise to motor commands to simulate motor response uncertainty
        
        Args:
            motor_commands: Motor RPM commands [4,]
            
        Returns:
            Noisy motor commands
        """
        std_dev = self.get_current_std_dev(NoiseType.MOTOR_NOISE.value)
        config = self.noise_configs[NoiseType.MOTOR_NOISE.value]
        
        if std_dev == 0.0:
            return motor_commands
            
        # 相对噪声（基于当前RPM的百分比）
        relative_noise = np.random.normal(config.mean, std_dev, 4)
        motor_noise = motor_commands * relative_noise
        
        # 应用裁剪
        if config.clip_range:
            # 裁剪相对噪声而不是绝对值
            relative_noise = np.clip(relative_noise, *config.clip_range)
            motor_noise = motor_commands * relative_noise
        
        noisy_commands = motor_commands + motor_noise
        
        # 确保RPM不为负
        noisy_commands = np.maximum(noisy_commands, 0)
        
        # 记录噪声历史
        self.noise_history[NoiseType.MOTOR_NOISE.value].append({
            'step': self.step_count,
            'motor_noise_norm': np.linalg.norm(motor_noise)
        })
        
        return noisy_commands

    def add_observation_noise(self, observation: np.ndarray) -> np.ndarray:
        """
        Add noise to the entire observation vector
        
        Args:
            observation: Full observation vector
            
        Returns:
            Noisy observation vector
        """
        std_dev = self.get_current_std_dev(NoiseType.OBSERVATION_NOISE.value)
        config = self.noise_configs[NoiseType.OBSERVATION_NOISE.value]
        
        if std_dev == 0.0:
            return observation
            
        # 为不同类型的观测添加不同强度的噪声
        obs_noise = np.random.normal(config.mean, std_dev, observation.shape)
        
        # 对不同观测维度应用不同的噪声强度
        # 位置(0-2): 标准噪声
        # 姿态(3-5): 标准噪声  
        # 速度(6-8): 标准噪声
        # 角速度(9-11): 更强噪声
        obs_noise[9:12] *= 2.0  # 角速度噪声更强
        
        # 电机历史(12-71): 较弱噪声
        if len(observation) > 12:
            obs_noise[12:72] *= 0.5
        
        # 激光雷达(72-83): 标准噪声
        # 目标信息(84-): 较弱噪声
        if len(observation) > 84:
            obs_noise[84:] *= 0.3
            
        # 应用裁剪
        if config.clip_range:
            obs_noise = np.clip(obs_noise, *config.clip_range)
        
        # 记录噪声历史
        self.noise_history[NoiseType.OBSERVATION_NOISE.value].append({
            'step': self.step_count,
            'obs_noise_norm': np.linalg.norm(obs_noise)
        })
        
        return observation + obs_noise

    def add_action_noise(self, action: np.ndarray) -> np.ndarray:
        """
        Add noise to actions before execution
        
        Args:
            action: Action vector (motor RPMs)
            
        Returns:
            Noisy action vector
        """
        std_dev = self.get_current_std_dev(NoiseType.ACTION_NOISE.value)
        config = self.noise_configs[NoiseType.ACTION_NOISE.value]
        
        if std_dev == 0.0:
            return action
            
        # 相对噪声（基于当前动作的百分比）
        relative_noise = np.random.normal(config.mean, std_dev, action.shape)
        action_noise = action * relative_noise
        
        # 应用裁剪
        if config.clip_range:
            relative_noise = np.clip(relative_noise, *config.clip_range)
            action_noise = action * relative_noise
        
        noisy_action = action + action_noise
        
        # 确保动作在合理范围内
        noisy_action = np.maximum(noisy_action, 0)
        
        # 记录噪声历史
        self.noise_history[NoiseType.ACTION_NOISE.value].append({
            'step': self.step_count,
            'action_noise_norm': np.linalg.norm(action_noise)
        })
        
        return noisy_action

    def add_attitude_noise(self, rpy: np.ndarray) -> np.ndarray:
        """
        Add noise to attitude measurements (roll, pitch, yaw)
        
        Args:
            rpy: Roll, pitch, yaw angles
            
        Returns:
            Noisy attitude angles
        """
        std_dev = self.get_current_std_dev(NoiseType.ATTITUDE_NOISE.value)
        config = self.noise_configs[NoiseType.ATTITUDE_NOISE.value]
        
        if std_dev == 0.0:
            return rpy
            
        attitude_noise = np.random.normal(config.mean, std_dev, 3)
        
        # 应用裁剪
        if config.clip_range:
            attitude_noise = np.clip(attitude_noise, *config.clip_range)
        
        # 记录噪声历史
        self.noise_history[NoiseType.ATTITUDE_NOISE.value].append({
            'step': self.step_count,
            'attitude_noise_norm': np.linalg.norm(attitude_noise)
        })
        
        return rpy + attitude_noise

    def step(self):
        """Update step counter for noise management"""
        self.step_count += 1

    def new_episode(self):
        """Update episode counter for noise management"""
        self.episode_count += 1
        
        # 清理过长的历史记录（保持最近1000步）
        for noise_type in self.noise_history:
            if len(self.noise_history[noise_type]) > 1000:
                self.noise_history[noise_type] = self.noise_history[noise_type][-1000:]

    def get_noise_statistics(self) -> Dict:
        """
        Get statistics about applied noise
        
        Returns:
            Dictionary with noise statistics for each type
        """
        stats = {}
        for noise_type, history in self.noise_history.items():
            if history:
                # 获取最近100步的统计信息
                recent_history = history[-100:] if len(history) > 100 else history
                
                # 计算不同指标的平均值
                if noise_type == NoiseType.SENSOR_NOISE.value:
                    avg_pos_noise = np.mean([h['pos_noise_norm'] for h in recent_history])
                    avg_vel_noise = np.mean([h['vel_noise_norm'] for h in recent_history])
                    avg_ang_vel_noise = np.mean([h['ang_vel_noise_norm'] for h in recent_history])
                    stats[noise_type] = {
                        'avg_position_noise': avg_pos_noise,
                        'avg_velocity_noise': avg_vel_noise,
                        'avg_angular_velocity_noise': avg_ang_vel_noise
                    }
                else:
                    # 其他噪声类型的通用统计
                    noise_key = [k for k in recent_history[0].keys() if 'noise' in k or 'force' in k][0]
                    avg_noise = np.mean([h[noise_key] for h in recent_history])
                    stats[noise_type] = {'avg_noise_magnitude': avg_noise}
                    
        return stats

    def print_noise_statistics(self):
        """Print current noise statistics"""
        stats = self.get_noise_statistics()
        print("\n📈 Noise Statistics (Recent 100 steps):")
        print("-" * 50)
        for noise_type, stat in stats.items():
            print(f"{noise_type}:")
            for key, value in stat.items():
                print(f"  {key}: {value:.4f}")
        print("-" * 50)

    def enable_noise_type(self, noise_type: Union[str, NoiseType]):
        """Enable a specific type of noise"""
        if isinstance(noise_type, NoiseType):
            noise_type = noise_type.value
        self.noise_configs[noise_type].enabled = True
        print(f"✅ Enabled {noise_type} noise")

    def disable_noise_type(self, noise_type: Union[str, NoiseType]):
        """Disable a specific type of noise"""
        if isinstance(noise_type, NoiseType):
            noise_type = noise_type.value
        self.noise_configs[noise_type].enabled = False
        print(f"❌ Disabled {noise_type} noise")

    def set_training_mode(self, enable_noise: bool):
        """
        Enable or disable all noise (useful for switching between training and evaluation)
        
        Args:
            enable_noise: Whether to enable noise
        """
        for config in self.noise_configs.values():
            config.enabled = enable_noise
        
        mode = "Training (with noise)" if enable_noise else "Evaluation (no noise)"
        print(f"🎯 Mode switched to: {mode}")


# 便利函数用于创建常用配置
def create_light_noise_manager(**kwargs) -> GaussianNoiseManager:
    """Create a noise manager with light noise settings"""
    manager = GaussianNoiseManager(**kwargs)
    
    # 减少所有噪声强度
    for noise_type in manager.noise_configs:
        manager.noise_configs[noise_type].std_dev *= 0.5
    
    print("🔇 Light noise mode activated")
    return manager


def create_heavy_noise_manager(**kwargs) -> GaussianNoiseManager:
    """Create a noise manager with heavy noise settings"""
    manager = GaussianNoiseManager(**kwargs)
    
    # 增加所有噪声强度
    for noise_type in manager.noise_configs:
        manager.noise_configs[noise_type].std_dev *= 2.0
    
    print("🔊 Heavy noise mode activated")
    return manager


def create_wind_focused_noise_manager(**kwargs) -> GaussianNoiseManager:
    """Create a noise manager focused on wind disturbances"""
    manager = GaussianNoiseManager(**kwargs)
    
    # 关闭大部分噪声，只保留风扰动
    for noise_type in manager.noise_configs:
        if noise_type != NoiseType.WIND_DISTURBANCE.value:
            manager.noise_configs[noise_type].enabled = False
    
    # 增强风扰动
    manager.noise_configs[NoiseType.WIND_DISTURBANCE.value].std_dev *= 1.5
    
    print("💨 Wind-focused noise mode activated")
    return manager
