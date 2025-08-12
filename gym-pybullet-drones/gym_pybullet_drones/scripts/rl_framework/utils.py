"""
Utility functions for the RL framework.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import argparse
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description='Deep Reinforcement Learning for Drone Control (Modular)')
    
    # Task parameters
    parser.add_argument('--task', default='unified', type=str,
                       choices=['unified'],  # 只支持 unified 任务
                       help='Task to train/evaluate (default: unified, only unified supported)')
    parser.add_argument('--trajectory_type', default='circle', type=str,
                       choices=['circle', 'figure8', 'waypoints'],
                       help='Trajectory type (kept for compatibility, not used)')
    
    # Training parameters
    parser.add_argument('--train_mode', default=True, type=str2bool,
                       help='Whether to train (True) or evaluate (False) (default: True)')
    parser.add_argument('--episodes', default=1000, type=int,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--learning_rate', default=3e-4, type=float,
                       help='Learning rate for training (default: 3e-4)')
    parser.add_argument('--eval_freq', default=2000, type=int,
                       help='Evaluation frequency during training (default: 2000)')
    parser.add_argument('--enable_obstacles', default=True, type=str2bool,
                       help='Whether to enable obstacles for unified task (default: True)')
    
    # Gaussian noise parameters
    parser.add_argument('--enable_noise', default=False, type=str2bool,
                       help='Whether to enable Gaussian noise during training (default: False)')
    parser.add_argument('--noise_level', default='medium', type=str,
                       choices=['light', 'medium', 'heavy'],
                       help='Noise level: light, medium, or heavy (default: medium)')
    parser.add_argument('--noise_decay', default=True, type=str2bool,
                       help='Whether noise should decay with training progress (default: True)')
    
    # Environment parameters
    parser.add_argument('--obs_type', default=ObservationType('kin'), type=ObservationType,
                       help='Observation type (default: kin)')
    parser.add_argument('--act_type', default=ActionType('rpm'), type=ActionType,
                       help='Action type (default: rpm)')
    parser.add_argument('--gui', default=True, type=str2bool,
                       help='Whether to use PyBullet GUI (default: True)')
    parser.add_argument('--record_video', default=False, type=str2bool,
                       help='Whether to record video (default: False)')
    parser.add_argument('--duration_sec', default=60, type=int,
                       help='Duration for evaluation in seconds (default: 60)')
    
    # Model parameters
    parser.add_argument('--load_model', default=None, type=str,
                       help='Path to model to load (for continuing training or evaluation)')
    parser.add_argument('--output_folder', default='results', type=str,
                       help='Output folder for logs and models (default: results)')
    parser.add_argument('--colab', default=False, type=str2bool,
                       help='Whether running in Colab (default: False)')
    
    # Utility parameters
    parser.add_argument('--list_models', action='store_true',
                       help='List all available trained models')
    
    # Performance evaluation parameters
    parser.add_argument('--performance_eval', default=False, action='store_true',
                       help='Enable performance evaluation mode')
    parser.add_argument('--evaluate_model', 
                        type=str, 
                        default='gym_pybullet_drones/scripts/results/unified/best_model.zip',
                        help='Path to model to evaluate (default: gym_pybullet_drones/scripts/results/unified/best_model.zip)')
    parser.add_argument('--eval_episodes', default=10, type=int,
                       help='Number of episodes for evaluation (default: 10)')
    parser.add_argument('--eval_duration', default=180, type=int,
                       help='Duration per episode in seconds for evaluation (default: 180)')
    parser.add_argument('--model_names', nargs='+', default=None,
                       help='Name for the model being evaluated')
    parser.add_argument('--generate_latex', default=False, type=str2bool,
                       help='Whether to generate LaTeX tables for academic papers (default: False)')
    parser.add_argument('--output_dir', default=None, type=str,
                       help='Custom output directory for performance evaluation results')
    
    return parser
