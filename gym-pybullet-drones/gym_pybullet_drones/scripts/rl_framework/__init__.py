"""
RL Framework for Drone Control

A modular framework for training and evaluating reinforcement learning agents
for drone control tasks.
"""

from .config import TrainingConfig, EnvironmentConfig, ModelConfig
from .notifications import NotificationManager
from .environment import EnvironmentFactory
from .model_manager import ModelManager
from .trainer import DroneRLTrainer
from .utils import create_argument_parser
from .gaussian_noise import (
    GaussianNoiseManager, 
    NoiseType, 
    NoiseConfig,
    create_light_noise_manager,
    create_heavy_noise_manager,
    create_wind_focused_noise_manager
)
from .performance_evaluation import DronePerformanceLogger, PerformanceMetrics

__version__ = "1.0.0"
__all__ = [
    "TrainingConfig",
    "EnvironmentConfig", 
    "ModelConfig",
    "NotificationManager",
    "EnvironmentFactory",
    "ModelManager", 
    "DroneRLTrainer",
    "create_argument_parser",
    "GaussianNoiseManager",
    "NoiseType",
    "NoiseConfig",
    "create_light_noise_manager",
    "create_heavy_noise_manager", 
    "create_wind_focused_noise_manager",
    "DronePerformanceLogger",
    "PerformanceMetrics"
]
