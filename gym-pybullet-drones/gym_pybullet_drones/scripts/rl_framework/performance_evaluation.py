#!/usr/bin/env python3
"""
Drone Performance Evaluation Module

This module provides comprehensive performance evaluation capabilities for drone RL models,
including data collection, analysis, and visualization of key performance metrics.

Author: GitHub Copilot Assistant
Date: August 12, 2025
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Try to import seaborn, use default matplotlib style if not available
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except ImportError:
    print("⚠️  Seaborn not available, using default matplotlib styling")
    plt.style.use('default')


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

@dataclass
class PerformanceMetrics:
    """Data class to store performance metrics."""
    
    # High Priority Metrics
    position_errors: List[float] = field(default_factory=list)
    task_completion_rate: float = 0.0
    task_completion_time: float = 0.0  # New metric: time to complete task
    control_energy: List[float] = field(default_factory=list)
    safety_events: List[Dict] = field(default_factory=list)
    
    # Medium Priority Metrics  
    attitude_stability: List[float] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    frequency_characteristics: Dict[str, List[float]] = field(default_factory=dict)
    robustness_scores: List[float] = field(default_factory=list)
    
    # Additional computed metrics
    rmse_position: float = 0.0
    max_position_error: float = 0.0
    steady_state_error: float = 0.0
    settling_time: float = 0.0
    overshoot: float = 0.0
    control_smoothness: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for saving."""
        result = {
            'position_errors': self.position_errors,
            'task_completion_rate': self.task_completion_rate,
            'task_completion_time': self.task_completion_time,
            'control_energy': self.control_energy,
            'safety_events': self.safety_events,
            'attitude_stability': self.attitude_stability,
            'response_times': self.response_times,
            'frequency_characteristics': self.frequency_characteristics,
            'robustness_scores': self.robustness_scores,
            'rmse_position': self.rmse_position,
            'max_position_error': self.max_position_error,
            'steady_state_error': self.steady_state_error,
            'settling_time': self.settling_time,
            'overshoot': self.overshoot,
            'control_smoothness': self.control_smoothness
        }
        
        # Add episode-level metrics if they exist
        if hasattr(self, 'episode_completion_rates'):
            result['episode_completion_rates'] = self.episode_completion_rates
        if hasattr(self, 'episode_completion_times'):
            result['episode_completion_times'] = self.episode_completion_times
        if hasattr(self, 'response_times_10_percent'):
            result['response_times_10_percent'] = self.response_times_10_percent
            
        return result


class DronePerformanceLogger:
    """
    Comprehensive performance evaluation logger for drone RL models.
    
    This class provides functionality to:
    1. Collect performance data during evaluation
    2. Compute key performance metrics
    3. Generate visualizations and reports
    """
    
    def __init__(self, output_folder: str = "results", logging_freq: int = 240):
        """
        Initialize the performance logger.
        
        Args:
            output_folder: Directory to save results
            logging_freq: Data logging frequency (Hz)
        """
        self.output_folder = output_folder
        self.logging_freq = logging_freq
        self.metrics = PerformanceMetrics()
        self.episode_data = []
        self.current_episode = {}
        
        # Create output directories
        self.performance_dir = os.path.join(output_folder, "performance_analysis")
        os.makedirs(self.performance_dir, exist_ok=True)
        
        # Target position for task completion evaluation
        self.target_position = np.array([0, 0, 1])  # Default hover target (will be updated from env)
        self.target_tolerance = 0.15  # meters (increased tolerance for realistic evaluation)
        self.required_hover_time = 3.0  # seconds
        
        # Data collection variables
        self.step_data = defaultdict(list)
        self.episode_start_time = None
        self.task_completed = False
        self.hover_start_time = None
        self.completion_time = 0.0  # Track individual episode completion time
        
        print(f"📊 Performance Logger initialized")
        print(f"   Output directory: {self.performance_dir}")
    
    def start_episode(self) -> None:
        """Start a new episode for data collection."""
        self.current_episode = {
            'step': 0,
            'positions': [],
            'velocities': [],
            'attitudes': [],
            'controls': [],
            'errors': [],
            'rewards': [],
            'timestamps': [],
            'safety_events': []
        }
        self.episode_start_time = time.time()
        self.task_completed = False
        self.hover_start_time = None
        self.completion_time = 0.0
        
    def update_target_position(self, info: Dict) -> None:
        """Update target position from environment info."""
        if 'target_pos' in info:
            self.target_position = np.array(info['target_pos'])
            print(f"🎯 Updated target position: {self.target_position}")
        
    def log_step(self, obs: np.ndarray, action: np.ndarray, reward: float, 
                 info: Dict, timestamp: float) -> None:
        """
        Log data for a single step.
        
        Args:
            obs: Observation from environment
            action: Action taken by agent
            reward: Reward received
            info: Additional information from environment
            timestamp: Current simulation time
        """
        if not hasattr(self, 'current_episode'):
            self.start_episode()
        
        # Update target position from environment info
        self.update_target_position(info)
        
        # Extract position, velocity, and attitude from observation
        if len(obs) >= 12:  # Kinematic observation
            position = obs[0:3]
            velocity = obs[3:6] if len(obs) >= 6 else np.zeros(3)
            attitude = obs[6:9] if len(obs) >= 9 else np.zeros(3)
        else:
            # Handle different observation formats
            position = obs[0:3] if len(obs) >= 3 else np.zeros(3)
            velocity = np.zeros(3)
            attitude = np.zeros(3)
        
        # Calculate position error
        position_error = np.linalg.norm(position - self.target_position)
        
        # Log step data
        self.current_episode['step'] += 1
        self.current_episode['positions'].append(position.copy())
        self.current_episode['velocities'].append(velocity.copy())
        self.current_episode['attitudes'].append(attitude.copy())
        self.current_episode['controls'].append(action.copy())
        self.current_episode['errors'].append(position_error)
        self.current_episode['rewards'].append(reward)
        self.current_episode['timestamps'].append(timestamp)
        
        # Check for task completion (sustained hover at target)
        if position_error <= self.target_tolerance:
            if self.hover_start_time is None:
                self.hover_start_time = timestamp
                print(f"📍 Started hovering at target (error: {position_error:.3f}m)")
            elif timestamp - self.hover_start_time >= self.required_hover_time and not self.task_completed:
                self.task_completed = True
                self.completion_time = timestamp
                print(f"✅ Task completed! Completion time: {self.completion_time:.2f}s")
        else:
            if self.hover_start_time is not None:
                print(f"⚠️  Left target area (error: {position_error:.3f}m)")
            self.hover_start_time = None
        
        # Check for safety events
        if position_error > 2.0:  # Large deviation
            self.current_episode['safety_events'].append({
                'type': 'large_deviation',
                'timestamp': timestamp,
                'error': position_error
            })
        
        if np.any(np.abs(attitude) > np.pi/3):  # Large attitude angle
            self.current_episode['safety_events'].append({
                'type': 'attitude_violation',
                'timestamp': timestamp,
                'attitude': attitude.copy()
            })
    
    def end_episode(self) -> None:
        """End current episode and compute metrics."""
        if not hasattr(self, 'current_episode'):
            return
        
        episode_duration = time.time() - self.episode_start_time
        
        # Compute episode metrics
        positions = np.array(self.current_episode['positions'])
        velocities = np.array(self.current_episode['velocities'])
        attitudes = np.array(self.current_episode['attitudes'])
        controls = np.array(self.current_episode['controls'])
        errors = np.array(self.current_episode['errors'])
        timestamps = np.array(self.current_episode['timestamps'])
        
        # High Priority Metrics
        self.metrics.position_errors.extend(errors.tolist())
        
        # Task completion metrics (per episode)
        if self.task_completed:
            # Task completed successfully
            completion_rate = 1.0
            completion_time = self.completion_time
            print(f"✅ Episode completed successfully in {completion_time:.2f}s")
        else:
            # Task not completed
            completion_rate = 0.0
            completion_time = episode_duration  # Use full episode duration as penalty
            print(f"❌ Episode failed to complete task (duration: {episode_duration:.2f}s)")
        
        # Store individual episode completion data
        if not hasattr(self.metrics, 'episode_completion_rates'):
            self.metrics.episode_completion_rates = []
        if not hasattr(self.metrics, 'episode_completion_times'):
            self.metrics.episode_completion_times = []
            
        self.metrics.episode_completion_rates.append(completion_rate)
        self.metrics.episode_completion_times.append(completion_time)
        
        # Update overall completion rate (average across all episodes)
        self.metrics.task_completion_rate = np.mean(self.metrics.episode_completion_rates)
        self.metrics.task_completion_time = np.mean(self.metrics.episode_completion_times)
        
        # Control energy
        if len(controls) > 0:
            control_energy = np.sum(np.linalg.norm(controls, axis=1))
            self.metrics.control_energy.append(control_energy)
        
        # Safety events
        self.metrics.safety_events.extend(self.current_episode['safety_events'])
        
        # Medium Priority Metrics
        if len(attitudes) > 0:
            # Attitude stability (standard deviation of attitude changes)
            attitude_changes = np.diff(attitudes, axis=0)
            attitude_stability = np.mean(np.std(attitude_changes, axis=0))
            self.metrics.attitude_stability.append(attitude_stability)
        
        # Response time (time to reach within target tolerance for the first time)
        first_within_tolerance = errors <= self.target_tolerance
        if np.any(first_within_tolerance):
            response_time = timestamps[np.where(first_within_tolerance)[0][0]]
            self.metrics.response_times.append(response_time)
            print(f"⏱️  Response time: {response_time:.2f}s")
        else:
            print(f"⚠️  Never reached target tolerance ({self.target_tolerance}m)")
        
        # Additional response time: time to reach within 10% of target distance
        initial_distance = np.linalg.norm(self.target_position)
        threshold_10_percent = 0.1 * initial_distance
        within_10_percent = errors <= threshold_10_percent
        if np.any(within_10_percent):
            response_time_10 = timestamps[np.where(within_10_percent)[0][0]]
            if not hasattr(self.metrics, 'response_times_10_percent'):
                self.metrics.response_times_10_percent = []
            self.metrics.response_times_10_percent.append(response_time_10)
        
        # Frequency characteristics (simplified - dominant frequency of position error)
        if len(errors) > 10:
            fft_errors = np.fft.fft(errors)
            freqs = np.fft.fftfreq(len(errors), 1/self.logging_freq)
            dominant_freq = freqs[np.argmax(np.abs(fft_errors[1:len(fft_errors)//2])) + 1]
            if 'position_error' not in self.metrics.frequency_characteristics:
                self.metrics.frequency_characteristics['position_error'] = []
            self.metrics.frequency_characteristics['position_error'].append(abs(dominant_freq))
        
        # Robustness score (inverse of error variance)
        if len(errors) > 1:
            error_variance = np.var(errors)
            robustness_score = 1.0 / (1.0 + error_variance)
            self.metrics.robustness_scores.append(robustness_score)
        
        # Store episode data
        self.episode_data.append({
            'duration': episode_duration,
            'positions': positions.tolist(),
            'errors': errors.tolist(),
            'controls': controls.tolist(),
            'task_completed': self.task_completed,
            'completion_time': completion_time,
            'completion_rate': completion_rate,
            'safety_events_count': len(self.current_episode['safety_events']),
            'target_position': self.target_position.tolist(),
            'final_position': positions[-1].tolist() if len(positions) > 0 else [0, 0, 0],
            'final_error': errors[-1] if len(errors) > 0 else float('inf')
        })
        
        # Compute additional metrics
        self._compute_additional_metrics()
        
    def _compute_additional_metrics(self) -> None:
        """Compute additional derived metrics."""
        if len(self.metrics.position_errors) > 0:
            self.metrics.rmse_position = np.sqrt(np.mean(np.square(self.metrics.position_errors)))
            self.metrics.max_position_error = np.max(self.metrics.position_errors)
            
            # Steady state error (error in last 20% of episode)
            last_20_percent = int(0.8 * len(self.metrics.position_errors))
            if last_20_percent < len(self.metrics.position_errors):
                self.metrics.steady_state_error = np.mean(self.metrics.position_errors[last_20_percent:])
        
        # Settling time (time to stay within 2% of target)
        # Simplified implementation - could be improved
        if len(self.metrics.response_times) > 0:
            self.metrics.settling_time = np.mean(self.metrics.response_times)
        
        # Control smoothness
        if len(self.metrics.control_energy) > 0:
            control_variations = np.diff(self.metrics.control_energy)
            if len(control_variations) > 0:
                self.metrics.control_smoothness = 1.0 / (1.0 + np.std(control_variations))
    
    def evaluate_model(self, model_path: str, num_episodes: int = 10, 
                      model_name: str = None, duration_sec: int = None) -> Dict:
        """
        Evaluate a model and collect performance metrics.
        
        Args:
            model_path: Path to the trained model
            num_episodes: Number of episodes to evaluate
            model_name: Name of the model for reporting
            duration_sec: Episode duration in seconds (default: 180 for proper evaluation)
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"🔍 Evaluating model: {model_name or model_path}")
        print(f"   Episodes: {num_episodes}")
        
        # Use longer duration for proper evaluation (same as GUI mode)
        eval_duration = duration_sec if duration_sec is not None else 180
        print(f"   Duration per episode: {eval_duration}s")
        
        # Reset metrics
        self.metrics = PerformanceMetrics()
        self.episode_data = []
        
        # Import necessary modules
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
        from stable_baselines3.common.env_util import make_vec_env
        from .environment import EnvironmentFactory
        from .config import EnvironmentConfig, TrainingConfig
        
        try:
            # 🔧 Fix model path handling - ensure correct absolute path
            # Check if the path is already absolute
            if not os.path.isabs(model_path):
                # If relative path, make it relative to the current working directory
                # Current working directory should be gym_pybullet_drones/gym_pybullet_drones/scripts
                model_path = os.path.abspath(model_path)
            
            # Ensure model file exists
            if not os.path.exists(model_path):
                # Try alternative relative path from scripts directory
                alt_path = os.path.join(os.getcwd(), 'results', 'unified', 'best_model.zip')
                if os.path.exists(alt_path):
                    model_path = alt_path
                    print(f"🔧 Found model at alternative path: {model_path}")
                else:
                    raise FileNotFoundError(f"Model file not found at {model_path} or {alt_path}")
            
            print(f"🔧 Using model path: {model_path}")
            
            # Find normalization stats path
            stats_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
            if not os.path.exists(stats_path):
                print(f"⚠️  VecNormalize stats not found at {stats_path}")
                print("   This may cause performance issues!")
                env_has_normalization = False
            else:
                print(f"✅ Found VecNormalize stats at {stats_path}")
                env_has_normalization = True
            
            # Load model
            model = PPO.load(model_path)
            
            # Create environment with same settings as GUI evaluation
            env_config = EnvironmentConfig(gui=False, duration_sec=eval_duration)
            training_config = TrainingConfig(task='unified', enable_obstacles=True)
            
            print(f"🔧 Environment Configuration:")
            print(f"   GUI: {env_config.gui}")
            print(f"   Duration: {env_config.duration_sec}s")
            print(f"   Obstacles: {training_config.enable_obstacles}")
            print(f"   VecNormalize: {env_has_normalization}")
            
            # Create environment kwargs (exactly like trainer.py load_test_model)
            env_kwargs = EnvironmentFactory.get_env_kwargs(
                training_config.task, 
                env_config, 
                training_config
            )
            env_kwargs.update({
                'gui': env_config.gui,
                'record': env_config.record_video,
                'randomize_init': True
            })
            
            # Debug: Print environment parameters
            print(f"🔍 Environment kwargs: {list(env_kwargs.keys())}")
            
            if env_has_normalization:
                # Create vectorized environment and apply normalization (exactly like trainer.py)
                env_raw = make_vec_env(
                    lambda: EnvironmentFactory.create_environment(
                        training_config.task, 
                        training_config.trajectory_type,
                        **env_kwargs
                    ),
                    n_envs=1
                )
                env = VecNormalize.load(stats_path, env_raw)
                env.training = False
                env.norm_reward = False
                print("✅ VecNormalize applied successfully!")
                print(f"   training: {env.training}, norm_reward: {env.norm_reward}, norm_obs: {env.norm_obs}")
                print(f"   obs_rms count: {env.obs_rms.count}, ret_rms count: {env.ret_rms.count}")
                print(f"   obs_rms mean: {env.obs_rms.mean[:5] if hasattr(env.obs_rms, 'mean') else 'N/A'}")
                print(f"   obs_rms var: {env.obs_rms.var[:5] if hasattr(env.obs_rms, 'var') else 'N/A'}")
            else:
                # Create raw environment without normalization
                env = EnvironmentFactory.create_environment(
                    training_config.task, 
                    training_config.trajectory_type,
                    **env_kwargs
                )
                print("⚠️  Using raw environment without normalization")
            
            # Calculate maximum steps based on duration (30 Hz control frequency)
            max_steps = eval_duration * 30  # 30 Hz control frequency
            print(f"   Maximum steps per episode: {max_steps}")
            
            # Run evaluation episodes
            for episode in range(num_episodes):
                print(f"   Episode {episode + 1}/{num_episodes}")
                
                self.start_episode()
                
                # Handle different environment types (vectorized vs raw)
                if env_has_normalization:
                    obs = env.reset()
                    # VecEnv returns obs as (n_envs, obs_dim) array
                    if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
                        obs = obs[0]  # Take first (and only) environment
                else:
                    result = env.reset()
                    # Handle potential tuple return from env.reset()
                    if isinstance(result, tuple):
                        obs, info = result
                    else:
                        obs = result
                
                done = False
                step_count = 0
                
                while not done and step_count < max_steps:
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # 🔧 Ensure action has correct shape for single drone (1, 4)
                    if action.ndim == 1:
                        action = action.reshape(1, -1)  # Convert (4,) to (1, 4)
                    
                    # Handle different environment types
                    if env_has_normalization:
                        # VecEnv returns arrays
                        new_obs, reward, done, info = env.step([action])
                        new_obs = new_obs[0]  # Take first environment
                        reward = reward[0]    # Take first environment
                        done = done[0]       # Take first environment
                        info = info[0]       # Take first environment
                    else:
                        # Raw environment - ensure action shape is correct for single drone
                        result = env.step(action)
                        # Handle different return formats from env.step()
                        if len(result) == 4:
                            new_obs, reward, done, info = result
                        elif len(result) == 5:
                            new_obs, reward, terminated, truncated, info = result
                            done = terminated or truncated
                        else:
                            raise ValueError(f"Unexpected return format from env.step(): {len(result)} values")
                    
                    # Debug: Print why episode ended early
                    if done and step_count < 100:  # If episode ends very early
                        print(f"⚠️  Episode ended early at step {step_count}")
                        if env_has_normalization:
                            print(f"   Done: {done}")
                        else:
                            print(f"   Terminated: {terminated}, Truncated: {truncated}")
                        if 'current_pos' in info:
                            print(f"   Final position: {info['current_pos']}")
                        if 'target_pos' in info:
                            print(f"   Target position: {info['target_pos']}")
                            distance = np.linalg.norm(info['current_pos'] - info['target_pos']) if 'current_pos' in info else 'unknown'
                            print(f"   Distance to target: {distance}")
                    
                    # Log step data
                    timestamp = step_count / self.logging_freq
                    self.log_step(obs, action, reward, info, timestamp)
                    
                    obs = new_obs
                    step_count += 1
                
                self.end_episode()
                
            env.close()
            
            # Generate results summary
            results = {
                'model_name': model_name or os.path.basename(model_path),
                'model_path': model_path,
                'num_episodes': num_episodes,
                'metrics': self.metrics.to_dict(),
                'evaluation_time': datetime.now().isoformat()
            }
            
            # Save results
            results_file = os.path.join(self.performance_dir, f"evaluation_results_{model_name or 'model'}.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            
            print(f"✅ Evaluation completed. Results saved to: {results_file}")
            return results
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def generate_performance_report(self) -> str:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Path to the generated report file
        """
        report_file = os.path.join(self.performance_dir, "performance_report.md")
        
        with open(report_file, 'w') as f:
            f.write("# Drone Performance Evaluation Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # High Priority Metrics
            f.write("## High Priority Metrics\n\n")
            f.write(f"- **Task Completion Rate**: {self.metrics.task_completion_rate:.2%}\n")
            f.write(f"- **Average Task Completion Time**: {self.metrics.task_completion_time:.2f} seconds\n")
            f.write(f"- **Position RMSE**: {self.metrics.rmse_position:.4f} m\n")
            f.write(f"- **Maximum Position Error**: {self.metrics.max_position_error:.4f} m\n")
            f.write(f"- **Average Control Energy**: {np.mean(self.metrics.control_energy):.2f}\n")
            f.write(f"- **Safety Events**: {len(self.metrics.safety_events)}\n\n")
            
            # Medium Priority Metrics
            f.write("## Medium Priority Metrics\n\n")
            f.write(f"- **Average Attitude Stability**: {np.mean(self.metrics.attitude_stability):.4f}\n")
            f.write(f"- **Average Response Time**: {np.mean(self.metrics.response_times):.2f} seconds\n")
            f.write(f"- **Average Robustness Score**: {np.mean(self.metrics.robustness_scores):.4f}\n")
            f.write(f"- **Control Smoothness**: {self.metrics.control_smoothness:.4f}\n")
            f.write(f"- **Steady State Error**: {self.metrics.steady_state_error:.4f} m\n\n")
            
            # Additional Statistics
            f.write("## Statistical Summary\n\n")
            if len(self.metrics.position_errors) > 0:
                f.write(f"- **Position Error Statistics**:\n")
                f.write(f"  - Mean: {np.mean(self.metrics.position_errors):.4f} m\n")
                f.write(f"  - Std: {np.std(self.metrics.position_errors):.4f} m\n")
                f.write(f"  - 95th Percentile: {np.percentile(self.metrics.position_errors, 95):.4f} m\n")
        
        print(f"📋 Performance report generated: {report_file}")
        return report_file
    
    def visualize_performance(self, save_plots: bool = True) -> None:
        """
        Generate comprehensive performance visualizations.
        
        Args:
            save_plots: Whether to save plots to files
        """
        print("📈 Generating performance visualizations...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Position Error Time Series
        ax1 = fig.add_subplot(gs[0, 0])
        if len(self.metrics.position_errors) > 0:
            ax1.plot(self.metrics.position_errors, 'b-', alpha=0.7, linewidth=1)
            ax1.axhline(y=self.target_tolerance, color='r', linestyle='--', label='Target Tolerance')
            ax1.set_title('Position Error Over Time', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Position Error (m)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Error Distribution Histogram
        ax2 = fig.add_subplot(gs[0, 1])
        if len(self.metrics.position_errors) > 0:
            ax2.hist(self.metrics.position_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(x=np.mean(self.metrics.position_errors), color='r', linestyle='--', label='Mean')
            ax2.set_title('Position Error Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Position Error (m)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Control Energy Over Episodes
        ax3 = fig.add_subplot(gs[0, 2])
        if len(self.metrics.control_energy) > 0:
            episodes = range(1, len(self.metrics.control_energy) + 1)
            ax3.plot(episodes, self.metrics.control_energy, 'go-', markersize=6)
            ax3.set_title('Control Energy per Episode', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Control Energy')
            ax3.grid(True, alpha=0.3)
        
        # 4. Task Completion Analysis
        ax4 = fig.add_subplot(gs[1, 0])
        if hasattr(self.metrics, 'episode_completion_rates') and len(self.metrics.episode_completion_rates) > 0:
            episodes = range(1, len(self.metrics.episode_completion_rates) + 1)
            ax4.bar(episodes, self.metrics.episode_completion_rates, alpha=0.7, color='green')
            ax4.set_title('Task Completion per Episode', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Completed (1=Yes, 0=No)')
            ax4.set_ylim(-0.1, 1.1)
            ax4.grid(True, alpha=0.3)
            
            # Add completion rate text
            overall_rate = np.mean(self.metrics.episode_completion_rates) * 100
            ax4.text(0.02, 0.98, f'Overall Rate: {overall_rate:.1f}%', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No Task Completion Data', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
            ax4.set_title('Task Completion per Episode', fontsize=12, fontweight='bold')
        
        # 5. Response Time Analysis
        ax5 = fig.add_subplot(gs[1, 1])
        if len(self.metrics.response_times) > 0:
            ax5.boxplot(self.metrics.response_times, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax5.set_title('Response Time Distribution', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Response Time (s)')
            ax5.grid(True, alpha=0.3)
            
            # Add statistics
            mean_response = np.mean(self.metrics.response_times)
            std_response = np.std(self.metrics.response_times)
            ax5.text(0.02, 0.98, f'Mean: {mean_response:.2f}s\nStd: {std_response:.2f}s\nSamples: {len(self.metrics.response_times)}', 
                    transform=ax5.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax5.text(0.5, 0.5, 'No Response Time Data', 
                    transform=ax5.transAxes, ha='center', va='center', fontsize=12)
            ax5.set_title('Response Time Distribution', fontsize=12, fontweight='bold')
        
        # 6. Attitude Stability
        ax6 = fig.add_subplot(gs[1, 2])
        if len(self.metrics.attitude_stability) > 0:
            episodes = range(1, len(self.metrics.attitude_stability) + 1)
            ax6.plot(episodes, self.metrics.attitude_stability, 'mo-', markersize=6)
            ax6.set_title('Attitude Stability per Episode', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Stability Score')
            ax6.grid(True, alpha=0.3)
        
        # 7. Robustness Scores
        ax7 = fig.add_subplot(gs[2, 0])
        if len(self.metrics.robustness_scores) > 0:
            episodes = range(1, len(self.metrics.robustness_scores) + 1)
            ax7.plot(episodes, self.metrics.robustness_scores, 'co-', markersize=6)
            ax7.set_title('Robustness Scores per Episode', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Episode')
            ax7.set_ylabel('Robustness Score')
            ax7.grid(True, alpha=0.3)
        
        # 8. Safety Events Summary
        ax8 = fig.add_subplot(gs[2, 1])
        safety_counts = [episode['safety_events_count'] for episode in self.episode_data]
        if safety_counts:
            episodes = range(1, len(safety_counts) + 1)
            ax8.bar(episodes, safety_counts, alpha=0.7, color='red')
            ax8.set_title('Safety Events per Episode', fontsize=12, fontweight='bold')
            ax8.set_xlabel('Episode')
            ax8.set_ylabel('Number of Events')
            ax8.grid(True, alpha=0.3)
        
        # 9. Performance Summary Radar Chart
        ax9 = fig.add_subplot(gs[2, 2], projection='polar')
        
        # Prepare data for radar chart
        metrics_names = ['Task Completion', 'Position Accuracy', 'Control Efficiency', 
                        'Attitude Stability', 'Response Speed', 'Robustness']
        
        # Normalize metrics to 0-1 scale
        metrics_values = []
        metrics_values.append(self.metrics.task_completion_rate)  # Task completion
        
        if self.metrics.rmse_position > 0:
            metrics_values.append(max(0, 1 - self.metrics.rmse_position))  # Position accuracy
        else:
            metrics_values.append(1.0)
        
        if len(self.metrics.control_energy) > 0:
            # Normalize control efficiency (lower energy is better)
            max_energy = max(self.metrics.control_energy) if self.metrics.control_energy else 1
            avg_energy = np.mean(self.metrics.control_energy)
            metrics_values.append(max(0, 1 - avg_energy / max_energy))
        else:
            metrics_values.append(0.5)
        
        if len(self.metrics.attitude_stability) > 0:
            avg_stability = np.mean(self.metrics.attitude_stability)
            metrics_values.append(min(1.0, avg_stability * 10))  # Scale up small stability values
        else:
            metrics_values.append(0.5)
        
        if len(self.metrics.response_times) > 0:
            avg_response = np.mean(self.metrics.response_times)
            metrics_values.append(max(0, 1 - avg_response / 10))  # Normalize assuming 10s max
        else:
            metrics_values.append(0.5)
        
        if len(self.metrics.robustness_scores) > 0:
            metrics_values.append(np.mean(self.metrics.robustness_scores))
        else:
            metrics_values.append(0.5)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        metrics_values += metrics_values[:1]  # Complete the circle
        
        ax9.plot(angles, metrics_values, 'o-', linewidth=2, color='blue')
        ax9.fill(angles, metrics_values, alpha=0.25, color='blue')
        ax9.set_xticks(angles[:-1])
        ax9.set_xticklabels(metrics_names)
        ax9.set_ylim(0, 1)
        ax9.set_title('Performance Summary', fontsize=12, fontweight='bold', y=1.08)
        ax9.grid(True)
        
        # 10. 3D Trajectory Plot (if episode data available)
        if self.episode_data and len(self.episode_data) > 0:
            ax10 = fig.add_subplot(gs[3, :], projection='3d')
            
            # Define colors for different episodes
            colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
            alphas = [0.8, 0.7, 0.7, 0.6, 0.6, 0.5]  # Gradually decrease alpha for later episodes
            
            # Plot trajectories from first 6 episodes (or all available if less than 6)
            num_episodes_to_plot = min(6, len(self.episode_data))
            
            for episode_idx in range(num_episodes_to_plot):
                positions = np.array(self.episode_data[episode_idx]['positions'])
                if len(positions) > 0:
                    color = colors[episode_idx % len(colors)]
                    alpha = alphas[episode_idx % len(alphas)]
                    
                    # Plot trajectory
                    ax10.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                             color=color, alpha=alpha, linewidth=2, 
                             label=f'Episode {episode_idx + 1}')
                    
                    # Plot start position for each episode
                    ax10.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 
                               c=color, s=40, marker='o', alpha=alpha)
                    
                    # Plot end position for each episode
                    ax10.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], 
                               c=color, s=40, marker='s', alpha=alpha)
            
            # Plot target position (only once)
            ax10.scatter([self.target_position[0]], [self.target_position[1]], [self.target_position[2]], 
                       c='red', s=150, marker='*', label='Target', edgecolors='black', linewidth=1)
            
            ax10.set_xlabel('X (m)')
            ax10.set_ylabel('Y (m)')
            ax10.set_zlabel('Z (m)')
            ax10.set_title(f'3D Flight Trajectories (First {num_episodes_to_plot} Episodes)', 
                          fontsize=12, fontweight='bold')
            ax10.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle('Comprehensive Drone Performance Analysis', fontsize=16, fontweight='bold')
        
        if save_plots:
            plot_file = os.path.join(self.performance_dir, "performance_analysis.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Performance plots saved to: {plot_file}")
            
            # Generate individual trajectory plots
            self._generate_individual_trajectory_plots()
        
        plt.show()
    
    def _generate_individual_trajectory_plots(self):
        """
        Generate a single figure with 6 subplot panels showing individual 3D trajectory plots for each episode.
        """
        if not self.episode_data or len(self.episode_data) == 0:
            print("⚠️  No episode data available for individual trajectory plots")
            return
        
        # Define colors for different episodes
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        # Generate plots for first 6 episodes (or all available if less than 6)
        num_episodes_to_plot = min(6, len(self.episode_data))
        
        print(f"📈 Generating combined trajectory plots for {num_episodes_to_plot} episodes...")
        
        # Create a figure with 2 rows and 3 columns of subplots
        fig = plt.figure(figsize=(18, 12))
        
        # Calculate global min/max ranges for consistent scaling across all subplots
        # Include both trajectory positions AND target positions
        all_positions = []
        all_targets = []
        
        for episode_idx in range(num_episodes_to_plot):
            positions = np.array(self.episode_data[episode_idx]['positions'])
            target_pos = np.array(self.episode_data[episode_idx]['target_position'])
            
            if len(positions) > 0:
                all_positions.append(positions)
            all_targets.append(target_pos.reshape(1, 3))  # Reshape to (1, 3) for consistent shape
        
        if all_positions:
            all_pos_combined = np.vstack(all_positions)
            all_targets_combined = np.vstack(all_targets)
            
            # Combine trajectory positions and target positions for global range calculation
            all_points = np.vstack([all_pos_combined, all_targets_combined])
            
            global_x_range = [all_points[:, 0].min(), all_points[:, 0].max()]
            global_y_range = [all_points[:, 1].min(), all_points[:, 1].max()]
            global_z_range = [all_points[:, 2].min(), all_points[:, 2].max()]
            
            # Add some padding
            x_padding = (global_x_range[1] - global_x_range[0]) * 0.1
            y_padding = (global_y_range[1] - global_y_range[0]) * 0.1
            z_padding = (global_z_range[1] - global_z_range[0]) * 0.1
            
            global_x_range = [global_x_range[0] - x_padding, global_x_range[1] + x_padding]
            global_y_range = [global_y_range[0] - y_padding, global_y_range[1] + y_padding]
            global_z_range = [global_z_range[0] - z_padding, global_z_range[1] + z_padding]
        
        for episode_idx in range(num_episodes_to_plot):
            positions = np.array(self.episode_data[episode_idx]['positions'])
            target_pos = np.array(self.episode_data[episode_idx]['target_position'])
            
            if len(positions) == 0:
                continue
                
            # Create subplot (2 rows, 3 columns)
            ax = fig.add_subplot(2, 3, episode_idx + 1, projection='3d')
            
            color = colors[episode_idx % len(colors)]
            
            # Plot trajectory
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   color=color, linewidth=2, label='Trajectory', alpha=0.8)
            
            # Plot start position
            ax.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 
                      c='green', s=60, marker='o', label='Start', 
                      edgecolors='black', linewidth=1)
            
            # Plot end position
            ax.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], 
                      c='orange', s=60, marker='s', label='End',
                      edgecolors='black', linewidth=1)
            
            # Plot target position (specific to this episode)
            ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], 
                      c='red', s=100, marker='*', label='Target', 
                      edgecolors='black', linewidth=1)
            
            # Set consistent ranges for all subplots
            if all_positions:
                ax.set_xlim(global_x_range)
                ax.set_ylim(global_y_range)
                ax.set_zlim(global_z_range)
            
            # Set labels and title
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_zlabel('Z (m)', fontsize=10)
            ax.set_title(f'Episode {episode_idx + 1}', 
                        fontsize=12, fontweight='bold', pad=10)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend (smaller and positioned better)
            ax.legend(loc='upper right', fontsize=8, markerscale=0.7)
            
            # Adjust tick label size
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='z', labelsize=8)
        
        # Add main title
        fig.suptitle('3D Flight Trajectories - Individual Episodes Comparison', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        # Save the combined plot
        trajectory_file = os.path.join(self.performance_dir, "trajectories_combined.png")
        plt.savefig(trajectory_file, dpi=300, bbox_inches='tight')
        print(f"📊 Combined trajectory plots saved to: {trajectory_file}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def generate_latex_table(self) -> str:
        """
        Generate LaTeX table for academic papers.
        
        Returns:
            Path to the generated LaTeX file
        """
        latex_file = os.path.join(self.performance_dir, "performance_table.tex")
        
        with open(latex_file, 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Drone Performance Evaluation Results}\n")
            f.write("\\label{tab:performance_results}\n")
            f.write("\\begin{tabular}{|l|c|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Metric} & \\textbf{Value} \\\\\n")
            f.write("\\hline\n")
            
            # High Priority Metrics
            f.write("\\multicolumn{2}{|l|}{\\textbf{High Priority Metrics}} \\\\\n")
            f.write("\\hline\n")
            f.write(f"Task Completion Rate & {self.metrics.task_completion_rate:.2%} \\\\\n")
            f.write(f"Task Completion Time & {self.metrics.task_completion_time:.2f} s \\\\\n")
            f.write(f"Position RMSE & {self.metrics.rmse_position:.4f} m \\\\\n")
            f.write(f"Max Position Error & {self.metrics.max_position_error:.4f} m \\\\\n")
            if len(self.metrics.control_energy) > 0:
                f.write(f"Average Control Energy & {np.mean(self.metrics.control_energy):.2f} \\\\\n")
            f.write(f"Safety Events & {len(self.metrics.safety_events)} \\\\\n")
            f.write("\\hline\n")
            
            # Medium Priority Metrics
            f.write("\\multicolumn{2}{|l|}{\\textbf{Medium Priority Metrics}} \\\\\n")
            f.write("\\hline\n")
            if len(self.metrics.attitude_stability) > 0:
                f.write(f"Average Attitude Stability & {np.mean(self.metrics.attitude_stability):.4f} \\\\\n")
            if len(self.metrics.response_times) > 0:
                f.write(f"Average Response Time & {np.mean(self.metrics.response_times):.2f} s \\\\\n")
            if len(self.metrics.robustness_scores) > 0:
                f.write(f"Average Robustness Score & {np.mean(self.metrics.robustness_scores):.4f} \\\\\n")
            f.write(f"Control Smoothness & {self.metrics.control_smoothness:.4f} \\\\\n")
            f.write(f"Steady State Error & {self.metrics.steady_state_error:.4f} m \\\\\n")
            f.write("\\hline\n")
            
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"📄 LaTeX table generated: {latex_file}")
        return latex_file


# Export the main class
__all__ = ['DronePerformanceLogger', 'PerformanceMetrics']
