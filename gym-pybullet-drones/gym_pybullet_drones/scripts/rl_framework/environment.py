"""
Environment factory for creating different types of drone environments.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from typing import Dict, Any
from gym_pybullet_drones.envs.DRLAviary import DRLAviary
from gym_pybullet_drones.utils.enums import ActionType

from .config import EnvironmentConfig, TrainingConfig


class EnvironmentFactory:
    """Factory for creating different types of environments."""
    
    TARGET_REWARDS = {
        "unified": {"default": 500000.0}
    }
    
    NETWORK_CONFIGS = {
        "complex": dict(net_arch=[dict(pi=[512, 512, 256], vf=[512, 512, 256])]) # for m1 Apple Silicon
        # "complex": dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])  # for GPU
    }
    
    @classmethod
    def get_target_reward(cls, task: str, act_type: ActionType) -> float:
        """Get target reward based on task and action type."""
        if task == "unified":
            return cls.TARGET_REWARDS["unified"]["default"]
        else:
            raise ValueError(f"Unknown task: {task}. Only 'unified' task is supported.")
    
    @classmethod
    def create_environment(cls, task: str, trajectory_type: str = "circle", **kwargs):
        """Create environment based on task type."""
        # 现在只支持 unified 任务
        if task == "unified":
            return DRLAviary(**kwargs)
        else:
            raise ValueError(f"Unknown task: {task}. Only 'unified' task is supported.")
    
    @classmethod
    def get_env_kwargs(cls, task: str, config: EnvironmentConfig, training_config: TrainingConfig) -> Dict[str, Any]:
        """Get environment-specific keyword arguments."""
        base_kwargs = {
            'obs': config.obs_type,
            'act': config.act_type,
            'gui': False,  # Training always uses False
            'record': False
        }
        
        # 只支持 unified 任务
        if task == "unified":
            unified_kwargs = {
                'num_obstacles': 8 if training_config.enable_obstacles else 0,
                'obstacle_radius': 0.25,
                'sensing_range': 2.0,
                'target_radius': 0.15,
                'episode_len_sec': 30,
                'randomize_init': True,
                'enable_obstacles': training_config.enable_obstacles
            }
            base_kwargs.update(unified_kwargs)
        else:
            raise ValueError(f"Unknown task: {task}. Only 'unified' task is supported.")
        
        return base_kwargs
    
    @classmethod
    def get_network_config(cls, task: str) -> Dict[str, Any]:
        """Get network configuration based on task complexity."""
        if task == "unified":
            return cls.NETWORK_CONFIGS["complex"]
        else:
            raise ValueError(f"Unknown task: {task}. Only 'unified' task is supported.")
