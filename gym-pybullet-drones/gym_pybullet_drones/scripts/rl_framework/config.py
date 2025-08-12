"""
Configuration classes for the RL framework.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from dataclasses import dataclass
from typing import Optional
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    task: str = "unified"  # 现在只支持 unified 任务
    trajectory_type: str = "circle"  # 保留但不使用
    episodes: int = 1000
    learning_rate: float = 3e-4
    eval_freq: int = 2000
    enable_obstacles: bool = True
    # 高斯噪声相关配置
    enable_noise: bool = False
    noise_level: str = "medium"  # "light", "medium", "heavy"
    noise_decay: bool = True     # 是否随训练进度衰减噪声
    

@dataclass
class EnvironmentConfig:
    """Environment configuration parameters."""
    obs_type: ObservationType = ObservationType('kin')
    act_type: ActionType = ActionType('rpm')
    gui: bool = True
    record_video: bool = False
    duration_sec: int = 30
    

@dataclass 
class NoiseConfig:
    """Gaussian noise configuration parameters."""
    enable_sensor_noise: bool = True
    enable_wind_disturbance: bool = True
    enable_motor_noise: bool = True
    enable_observation_noise: bool = True
    enable_action_noise: bool = True
    enable_attitude_noise: bool = True
    sensor_noise_std: float = 0.02
    wind_noise_std: float = 0.5
    motor_noise_std: float = 0.05
    observation_noise_std: float = 0.01
    action_noise_std: float = 0.02
    attitude_noise_std: float = 0.05
    

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    load_model: Optional[str] = None
    output_folder: str = 'results'
    colab: bool = False
