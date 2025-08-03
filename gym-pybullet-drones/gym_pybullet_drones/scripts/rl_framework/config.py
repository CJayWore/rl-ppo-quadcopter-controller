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
    

@dataclass
class EnvironmentConfig:
    """Environment configuration parameters."""
    obs_type: ObservationType = ObservationType('kin')
    act_type: ActionType = ActionType('rpm')
    gui: bool = True
    record_video: bool = False
    duration_sec: int = 30
    

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    load_model: Optional[str] = None
    output_folder: str = 'results'
    colab: bool = False
