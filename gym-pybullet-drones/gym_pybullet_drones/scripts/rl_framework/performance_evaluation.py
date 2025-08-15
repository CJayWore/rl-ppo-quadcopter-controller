#!/usr/bin/env python3
"""
Drone Performance Evaluation Module

This module provides comprehensive performance evaluation capabilities for drone RL models,
including data collection, analysis, and visualization of key performance metrics.
Uses Logger.py data structure for consistent data collection while providing advanced visualization.

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

# Add gym wrapper import
import gymnasium as gym

class ActionReshapeWrapper(gym.Wrapper):
    """Wrapper to reshape actions for VecEnv compatibility with multi-drone environments."""
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        # Ensure action has the correct shape for multi-drone environment
        if isinstance(action, np.ndarray) and action.ndim == 1:
            # Reshape from (action_dim,) to (1, action_dim) for single drone
            action = action.reshape(1, -1)
        return self.env.step(action)

# Import Logger class for data collection
try:
    from ...utils.Logger import Logger
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'utils')
    sys.path.insert(0, utils_path)
    from Logger import Logger

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


class DronePerformanceLogger(Logger):
    """
    Comprehensive performance evaluation logger for drone RL models.
    
    This class extends Logger to provide advanced performance analysis while using
    the same data collection structure. It provides functionality to:
    1. Collect performance data using Logger's proven data structure
    2. Compute key performance metrics
    3. Generate advanced visualizations and reports
    """
    
    def __init__(self, output_folder: str = "results", logging_freq: int = 240, 
                 num_drones: int = 1, duration_sec: int = 0):
        """
        Initialize the performance logger.
        
        Args:
            output_folder: Directory to save results
            logging_freq: Data logging frequency (Hz)
            num_drones: Number of drones to track
            duration_sec: Expected duration for preallocation
        """
        # Initialize parent Logger class
        super().__init__(logging_freq, output_folder, num_drones, duration_sec)
        
        self.metrics = PerformanceMetrics()
        self.episode_data = []
        self.current_episode_idx = 0
        
        # Create performance analysis directory
        self.performance_dir = os.path.join(output_folder, "performance_analysis")
        os.makedirs(self.performance_dir, exist_ok=True)
        
        # Store the actual final position from the last step before episode ends
        # This prevents contamination from environment reset operations
        self.last_recorded_position = None
        self.last_recorded_error = None
        
        # Target position for task completion evaluation
        self.target_position = np.array([3, 3, 1.5])  # Default target matching DRLAviary
        self.target_tolerance = 0.15  # meters
        self.required_hover_time = 3.0  # seconds
        
        # Episode tracking variables
        self.episode_start_time = None
        self.episode_start_counter = 0
        self.task_completed = False
        self.hover_start_time = None
        self.completion_time = 0.0
        self.target_updated_this_episode = False
        
        print(f"Performance Logger initialized (extends Logger)")
        print(f"   Output directory: {self.performance_dir}")
        print(f"   Logging frequency: {logging_freq} Hz")
        print(f"   Tracking {num_drones} drone(s)")

    def start_episode(self, drone_id: int = 0) -> None:
        """Start a new episode for data collection."""
        self.episode_start_time = time.time()
        self.episode_start_counter = int(self.counters[drone_id])
        self.task_completed = False
        self.hover_start_time = None
        self.completion_time = 0.0
        self.current_episode_idx += 1
        self.target_updated_this_episode = False  # Flag to track target position updates
        
        print(f"Episode {self.current_episode_idx} started for drone {drone_id}")

    def update_target_position(self, info: Dict) -> None:
        """Update target position from environment info."""
        if 'target_pos' in info and not self.target_updated_this_episode:
            self.target_position = np.array(info['target_pos'])
            self.target_updated_this_episode = True
            print(f"Updated target position: {self.target_position}")

    def log_step_with_performance(self, drone: int, timestamp: float, state: np.ndarray,
                                 action: np.ndarray, reward: float, info: Dict, 
                                 control: np.ndarray = None) -> None:
        """
        Enhanced log step that combines Logger's data collection with performance tracking.
        
        Args:
            drone: Drone ID
            timestamp: Simulation timestamp
            state: Full state array (20 elements as per Logger)
            action: Action taken by agent (becomes RPM in state)
            reward: Reward received
            info: Additional information from environment
            control: Control target (12 elements as per Logger)
        """
        # Use default control array if not provided
        if control is None:
            control = np.zeros(12)
        
        # Update target position from environment info
        self.update_target_position(info)
        
        # Use parent Logger's log method for consistent data structure
        self.log(drone, timestamp, state, control)
        
        # Extract position for performance evaluation
        # Logger stores states as: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw, ang_vel_x, ang_vel_y, ang_vel_z, rpm0, rpm1, rpm2, rpm3]
        current_counter = int(self.counters[drone]) - 1  # Get the counter that was just logged
        if current_counter >= 0:
            # Extract position from logged state
            position = self.states[drone, 0:3, current_counter]
            velocity = self.states[drone, 3:6, current_counter] 
            attitude = self.states[drone, 6:9, current_counter]
            
            # Calculate position error
            position_error = np.linalg.norm(position - self.target_position)
            
            # Store the last recorded position and error for each episode
            self.last_recorded_position = position.copy()
            self.last_recorded_error = position_error
            
            # Debug logging removed for production
            
            # Track task completion (sustained hover at target)
            if position_error <= self.target_tolerance:
                if self.hover_start_time is None:
                    self.hover_start_time = timestamp
                    print(f"Started hovering at target (error: {position_error:.3f}m)")
                elif timestamp - self.hover_start_time >= self.required_hover_time and not self.task_completed:
                    self.task_completed = True
                    self.completion_time = timestamp
                    print(f"Task completed! Completion time: {self.completion_time:.2f}s")
            else:
                if self.hover_start_time is not None:
                    print(f"⚠️  Left target area (error: {position_error:.3f}m)")
                self.hover_start_time = None
            
            # Check for safety events
            safety_events = []
            if position_error > 2.0:  # Large deviation
                safety_events.append({
                    'type': 'large_deviation',
                    'timestamp': timestamp,
                    'error': position_error
                })
            
            if np.any(np.abs(attitude) > np.pi/3):  # Large attitude angle
                safety_events.append({
                    'type': 'attitude_violation', 
                    'timestamp': timestamp,
                    'attitude': attitude.copy()
                })
            
            # Store safety events for this step (will be aggregated in end_episode)
            if not hasattr(self, 'current_safety_events'):
                self.current_safety_events = []
            self.current_safety_events.extend(safety_events)
    
    def end_episode(self, drone_id: int = 0) -> None:
        """End current episode and compute metrics using Logger data."""
        if not hasattr(self, 'episode_start_time') or self.episode_start_time is None:
            print("⚠️  No episode started, cannot end episode")
            return
        
        episode_duration = time.time() - self.episode_start_time
        current_counter = int(self.counters[drone_id])
        episode_length = current_counter - self.episode_start_counter
        
        print(f"Ending episode {self.current_episode_idx} (duration: {episode_duration:.2f}s, {episode_length} steps)")
        
        if episode_length <= 0:
            print("⚠️  No data logged for this episode")
            return
            
        # Extract data from Logger's data structure for this episode ONLY
        # Make sure we don't include any data from the next episode
        start_idx = self.episode_start_counter
        end_idx = current_counter  # This should be exclusive (current_counter points to next available slot)
        
        # Double-check that we have valid data range
        if end_idx <= start_idx:
            print("⚠️  Invalid data range for episode")
            return
        
        # Get episode data from Logger arrays
        timestamps = self.timestamps[drone_id, start_idx:end_idx]
        states = self.states[drone_id, :, start_idx:end_idx]  # Shape: (16, episode_length)
        controls = self.controls[drone_id, :, start_idx:end_idx] # Shape: (12, episode_length)
        
        # Extract positions, velocities, attitudes from states
        # Logger state order: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw, ang_vel_x, ang_vel_y, ang_vel_z, rpm0, rpm1, rpm2, rpm3]
        positions = states[0:3, :].T  # Shape: (episode_length, 3)
        velocities = states[3:6, :].T  # Shape: (episode_length, 3)
        attitudes = states[6:9, :].T   # Shape: (episode_length, 3)
        rpms = states[12:16, :].T      # Shape: (episode_length, 4)
        
        # Debug logging removed for production
        
        # Calculate position errors
        errors = np.array([np.linalg.norm(pos - self.target_position) for pos in positions])
        
        # High Priority Metrics
        self.metrics.position_errors.extend(errors.tolist())
        
        # Task completion metrics
        if self.task_completed:
            completion_rate = 1.0
            completion_time = self.completion_time
            print(f"Episode completed successfully in {completion_time:.2f}s")
        else:
            completion_rate = 0.0
            completion_time = episode_duration
            print(f"❌ Episode failed to complete task (duration: {episode_duration:.2f}s)")
        
        # Store episode completion data
        if not hasattr(self.metrics, 'episode_completion_rates'):
            self.metrics.episode_completion_rates = []
        if not hasattr(self.metrics, 'episode_completion_times'):
            self.metrics.episode_completion_times = []
            
        self.metrics.episode_completion_rates.append(completion_rate)
        self.metrics.episode_completion_times.append(completion_time)
        
        # Update overall completion rate
        self.metrics.task_completion_rate = np.mean(self.metrics.episode_completion_rates)
        self.metrics.task_completion_time = np.mean(self.metrics.episode_completion_times)
        
        # Control energy (using RPM data)
        if len(rpms) > 0:
            # Calculate control energy as sum of squared RPM values
            control_energy = np.sum(np.sum(rpms**2, axis=1))
            self.metrics.control_energy.append(control_energy)
        
        # Safety events
        if hasattr(self, 'current_safety_events'):
            self.metrics.safety_events.extend(self.current_safety_events)
            safety_events_count = len(self.current_safety_events)
            self.current_safety_events = []  # Reset for next episode
        else:
            safety_events_count = 0
        
        # Attitude stability (standard deviation of attitude angles)
        if len(attitudes) > 0:
            attitude_stability = 1.0 / (1.0 + np.mean(np.std(attitudes, axis=0)))
            self.metrics.attitude_stability.append(attitude_stability)
        
        # Response time (time to reach within target tolerance for first time)
        first_within_tolerance = errors <= self.target_tolerance
        if np.any(first_within_tolerance):
            first_success_idx = np.where(first_within_tolerance)[0][0]
            response_time = timestamps[first_success_idx] - timestamps[0] if len(timestamps) > first_success_idx else 0
            self.metrics.response_times.append(response_time)
        else:
            self.metrics.response_times.append(episode_duration)  # Use full duration if never reached
        
        # Frequency characteristics (dominant frequency of position error)
        if len(errors) > 10:
            try:
                fft_errors = np.fft.fft(errors)
                freqs = np.fft.fftfreq(len(errors), 1/self.LOGGING_FREQ_HZ)
                dominant_freq = freqs[np.argmax(np.abs(fft_errors[1:len(fft_errors)//2])) + 1]
                if 'position_error' not in self.metrics.frequency_characteristics:
                    self.metrics.frequency_characteristics['position_error'] = []
                self.metrics.frequency_characteristics['position_error'].append(abs(dominant_freq))
            except:
                pass  # Skip if FFT fails
        
        # Robustness score (inverse of error variance)
        if len(errors) > 1:
            error_variance = np.var(errors)
            robustness_score = 1.0 / (1.0 + error_variance)
            self.metrics.robustness_scores.append(robustness_score)
        
        # Store episode data
        # Use the last recorded position to avoid contamination from env reset
        actual_final_position = self.last_recorded_position.tolist() if self.last_recorded_position is not None else (positions[-1].tolist() if len(positions) > 0 else [0, 0, 0])
        actual_final_error = self.last_recorded_error if self.last_recorded_error is not None else (errors[-1] if len(errors) > 0 else float('inf'))
        
        episode_summary = {
            'episode_id': self.current_episode_idx,
            'duration': episode_duration,
            'step_count': episode_length,
            'positions': positions.tolist(),
            'velocities': velocities.tolist(),
            'attitudes': attitudes.tolist(),
            'controls': rpms.tolist(),  # Store RPM as controls for visualization
            'errors': errors.tolist(),
            'timestamps': timestamps.tolist(),
            'task_completed': self.task_completed,
            'completion_time': completion_time,
            'completion_rate': completion_rate,
            'safety_events_count': safety_events_count,
            'target_position': self.target_position.tolist(),
            'final_position': actual_final_position,
            'final_error': actual_final_error,
            'mean_error': np.mean(errors) if len(errors) > 0 else float('inf'),
            'max_error': np.max(errors) if len(errors) > 0 else float('inf'),
            'control_energy': np.sum(np.sum(rpms**2, axis=1)) if len(rpms) > 0 else 0,
        }
        
        self.episode_data.append(episode_summary)
        
        # Compute additional metrics
        self._compute_additional_metrics()
        
        print(f"   Final error: {episode_summary['final_error']:.3f}m")
        print(f"   Mean error: {episode_summary['mean_error']:.3f}m")
        print(f"   Control energy: {episode_summary['control_energy']:.2f}")
        print(f"   Safety events: {safety_events_count}")

    def _compute_additional_metrics(self) -> None:
        """Compute additional derived metrics."""
        if len(self.metrics.position_errors) > 0:
            self.metrics.rmse_position = np.sqrt(np.mean(np.array(self.metrics.position_errors)**2))
            self.metrics.max_position_error = np.max(self.metrics.position_errors)
            self.metrics.steady_state_error = np.mean(self.metrics.position_errors[-10:]) if len(self.metrics.position_errors) >= 10 else np.mean(self.metrics.position_errors)
        
        # Settling time (simplified implementation)
        if len(self.metrics.response_times) > 0:
            self.metrics.settling_time = np.mean(self.metrics.response_times)
        
        # Control smoothness
        if len(self.metrics.control_energy) > 0:
            energy_values = np.array(self.metrics.control_energy)
            if len(energy_values) > 1:
                self.metrics.control_smoothness = 1.0 / (1.0 + np.std(energy_values))
    
    def evaluate_model(self, model_path: str, num_episodes: int = 10, 
                      model_name: str = None, duration_sec: int = None, gui_enabled: bool = False) -> Dict:
        """
        Evaluate a model and collect performance metrics.
        
        Args:
            model_path: Path to the trained model
            num_episodes: Number of episodes to evaluate
            model_name: Name of the model for reporting
            duration_sec: Episode duration in seconds (default: 180 for proper evaluation)
            gui_enabled: Whether to enable GUI visualization during evaluation
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"Evaluating model: {model_name or model_path}")
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
            # Fix model path handling - ensure correct absolute path
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
                else:
                    raise FileNotFoundError(f"Model file not found at {model_path} or {alt_path}")
            
            # Find normalization stats path
            stats_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
            if not os.path.exists(stats_path):
                env_has_normalization = False
            else:
                env_has_normalization = True
            
            # Load model
            model = PPO.load(model_path)
            
            # Create environment with GUI support
            env_config = EnvironmentConfig(gui=gui_enabled, duration_sec=eval_duration)
            training_config = TrainingConfig(task='unified', enable_obstacles=True)
            
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
            
            if env_has_normalization:
                # Create vectorized environment and apply normalization (exactly like trainer.py)
                env_raw = make_vec_env(
                    lambda: ActionReshapeWrapper(EnvironmentFactory.create_environment(
                        training_config.task, 
                        training_config.trajectory_type,
                        **env_kwargs
                    )),
                    n_envs=1
                )
                env = VecNormalize.load(stats_path, env_raw)
                env.training = False
                env.norm_reward = False
            else:
                # Create raw environment without normalization
                env = EnvironmentFactory.create_environment(
                    training_config.task, 
                    training_config.trajectory_type,
                    **env_kwargs
                )
            
            # Calculate maximum steps based on duration (30 Hz control frequency)
            max_steps = eval_duration * 30  # 30 Hz control frequency
            
            # Run evaluation episodes
            for episode in range(num_episodes):
                
                # Start episode tracking BEFORE environment reset
                self.start_episode(drone_id=0)  # Start episode for drone 0
                
                # Handle different environment types (vectorized vs raw)
                if env_has_normalization:
                    obs = env.reset()
                    # VecEnv returns obs as (n_envs, obs_dim) array
                    if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
                        obs = obs[0]  # Get first environment's observation
                    
                    # Get initial info to update target position immediately
                    # For VecEnv, we need to step once to get info
                    action, _ = model.predict(obs, deterministic=True)
                    
                    obs, reward, done, info = env.step(action)
                    obs = obs[0] if isinstance(obs, np.ndarray) and len(obs.shape) > 1 else obs
                    info = info[0] if isinstance(info, list) and len(info) > 0 else info
                    done = done[0] if isinstance(done, np.ndarray) else done
                    
                    # Update target position from first step info
                    self.update_target_position(info)
                    
                    step_count = 1  # We've already taken one step
                else:
                    obs = env.reset()
                    result = obs
                    # Handle potential tuple return from env.reset()
                    if isinstance(result, tuple):
                        obs, info = result
                        # Update target position from reset info
                        self.update_target_position(info)
                    else:
                        obs = result
                        # Get info from first step for raw environment
                        action, _ = model.predict(obs, deterministic=True)
                        # For raw environment, ensure action is 1D for single drone
                        if action.ndim > 1:
                            action = action[0]  # Take first drone's action (convert from (1,4) to (4,))
                        obs, reward, terminated, truncated, info = env.step(action)
                        # Update target position from first step info
                        self.update_target_position(info)
                        done = terminated or truncated
                        step_count = 1  # We've already taken one step
                        if done:
                            print(f"⚠️  Episode ended immediately after reset/first step")
                            continue
                
                if 'step_count' not in locals():
                    step_count = 0
                    done = False
                
                while not done and step_count < max_steps:
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Handle different environment types
                    if env_has_normalization:
                        # VecEnv - use action directly as in trainer.py
                        new_obs, reward, done, info = env.step(action)
                        new_obs = new_obs[0]  # Take first environment
                        reward = reward[0]    # Take first environment
                        done = done[0]       # Take first environment
                        info = info[0]       # Take first environment
                    else:
                        # Raw environment - action should be 1D for single drone 
                        if action.ndim > 1:
                            action = action[0]  # Take first drone's action (convert from (1,4) to (4,))
                        result = env.step(action)
                        # Handle different return formats from env.step()
                        if len(result) == 4:
                            new_obs, reward, done, info = result
                        elif len(result) == 5:
                            new_obs, reward, terminated, truncated, info = result
                            done = terminated or truncated
                        else:
                            raise ValueError(f"Unexpected return format from env.step(): {len(result)} values")
                    
                    # Prepare state array for Logger (20 elements as expected)
                    # Convert observation to full state format expected by Logger
                    timestamp = step_count / self.LOGGING_FREQ_HZ
                    
                    # Extract current drone state from environment info or obs
                    if 'current_state' in info and len(info['current_state']) >= 20:
                        # Use full state from environment if available
                        state = np.array(info['current_state'][:20])
                    else:
                        # Construct state from available observation and info
                        state = np.zeros(20)
                        if len(obs) >= 3:  # Position
                            state[0:3] = obs[0:3]
                        if len(obs) >= 6:  # Velocity  
                            state[10:13] = obs[3:6]
                        if len(obs) >= 9:  # Attitude
                            state[7:10] = obs[6:9]
                        if len(obs) >= 12:  # Angular velocity
                            state[13:16] = obs[9:12]
                        
                        # Add RPM from action (scaled to realistic values)
                        if action.size >= 4:
                            action_flat = action.flatten()
                            # Scale normalized action [-1,1] to realistic RPM range [0, 20000]
                            rpm_values = (action_flat + 1) * 10000  # Map [-1,1] to [0, 20000]
                            state[16:20] = rpm_values[:4]
                    
                    # Get actual drone state vector (20 elements) from the environment
                    # For VecEnv, we need to access the underlying environment correctly
                    if env_has_normalization:
                        try:
                            # Try different access paths to get the underlying DRLAviary instance
                            underlying_env = None
                            
                            # Method 1: Through ActionReshapeWrapper
                            if hasattr(env, 'envs') and hasattr(env.envs[0], 'env'):
                                wrapper_env = env.envs[0].env
                                if hasattr(wrapper_env, 'env'):  # ActionReshapeWrapper.env
                                    underlying_env = wrapper_env.env
                                elif hasattr(wrapper_env, '_getDroneStateVector'):
                                    underlying_env = wrapper_env
                            
                            # Method 2: Direct access if Method 1 failed
                            if underlying_env is None and hasattr(env, 'envs'):
                                if hasattr(env.envs[0], '_getDroneStateVector'):
                                    underlying_env = env.envs[0]
                            
                            if underlying_env and hasattr(underlying_env, '_getDroneStateVector'):
                                # Success! Get raw state vector (20 elements) - true physical state
                                state = underlying_env._getDroneStateVector(0)
                                timestamp = step_count / underlying_env.CTRL_FREQ
                                
                                # Debug: Print position comparison every 30 steps
                                if step_count % 30 == 0:  
                                    true_pos = state[0:3]
                                    # Denormalize observations to get physical coordinates
                                    if hasattr(env, 'unnormalize_obs'):
                                        try:
                                            denorm_obs = env.unnormalize_obs(new_obs)
                                            obs_pos = denorm_obs[:3] if len(denorm_obs) >= 3 else np.array([0, 0, 0])
                                        except:
                                            obs_pos = new_obs[:3] if len(new_obs) >= 3 else np.array([0, 0, 0])
                                    else:
                                        obs_pos = new_obs[:3] if len(new_obs) >= 3 else np.array([0, 0, 0])
                            else:
                                # Fallback: use observation data
                                state = np.zeros(20)
                                if hasattr(env, 'unnormalize_obs'):
                                    try:
                                        denorm_obs = env.unnormalize_obs(new_obs)
                                        state[0:3] = denorm_obs[:3] if len(denorm_obs) >= 3 else np.array([0, 0, 0])
                                    except:
                                        state[0:3] = new_obs[:3] if len(new_obs) >= 3 else np.array([0, 0, 0])
                                else:
                                    state[0:3] = new_obs[:3] if len(new_obs) >= 3 else np.array([0, 0, 0])
                                timestamp = step_count * 0.033
                                
                        except Exception as e:
                            print(f"⚠️  Exception accessing state vector: {e}")
                            state = np.zeros(20)
                            state[0:3] = new_obs[:3] if len(new_obs) >= 3 else np.array([0, 0, 0])
                            timestamp = step_count * 0.033
                    else:
                        # For raw environment (non-normalized)
                        if hasattr(env, '_getDroneStateVector'):
                            state = env._getDroneStateVector(0)
                            timestamp = step_count / env.CTRL_FREQ
                        else:
                            # Fallback: construct state from observations
                            state = np.zeros(20)
                            state[0:3] = new_obs[:3] if len(new_obs) >= 3 else np.array([0, 0, 0])
                            timestamp = step_count * 0.033
                    
                    # Prepare control array (12 elements)
                    control = np.zeros(12)
                    
                    # Check if state is corrupted BEFORE logging
                    if (state[0] == 0.0 and state[1] == 0.0 and state[2] == 1.0):
                        break  # Stop immediately when we detect reset state
                        
                    # Save state BEFORE calling Logger methods to avoid corruption
                    # Store the true final state before any Logger operations
                    true_final_position = state[0:3].copy() if len(state) >= 3 else np.array([0, 0, 0])
                    true_final_error = np.linalg.norm(true_final_position - self.target_position)
                    
                    # Use enhanced logging method
                    self.log_step_with_performance(
                        drone=0, 
                        timestamp=timestamp, 
                        state=state,
                        action=action.flatten(), 
                        reward=reward, 
                        info=info,
                        control=control
                    )
                    
                    # Store the clean position data
                    self.last_recorded_position = true_final_position.copy()
                    self.last_recorded_error = true_final_error
                    
                    obs = new_obs
                    step_count += 1
                    
                    # Check if this will be the last step of the episode
                    if done or step_count >= max_steps:
                        break  # Exit the loop immediately to preserve state
                
                # End episode immediately when done, before any reset operations
                self.end_episode(drone_id=0)  # End episode for drone 0
                
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
            
            print(f"Evaluation completed. Results saved to: {results_file}")
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
        
        print(f"Performance report generated: {report_file}")
        return report_file
    
    def visualize_performance(self, save_plots: bool = True) -> None:
        """
        Generate comprehensive performance visualizations.
        
        Args:
            save_plots: Whether to save plots to files
        """
        print("Generating performance visualizations...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, hspace=0.3, wspace=0.3)
        
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
        
        plt.suptitle('Comprehensive Drone Performance Analysis', fontsize=16, fontweight='bold')
        
        if save_plots:
            plot_file = os.path.join(self.performance_dir, "performance_analysis.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Performance plots saved to: {plot_file}")
            
            # Generate individual trajectory plots
            self._generate_individual_trajectory_plots()
        
        plt.show()
    
    def _generate_individual_trajectory_plots(self):
        """
        Generate a single figure with 6 subplot panels showing individual 3D trajectory plots for each episode.
        """
        if not self.episode_data or len(self.episode_data) == 0:
            print("No episode data available for individual trajectory plots")
            return
        
        # Define colors for different episodes
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        # Generate plots for first 6 episodes (or all available if less than 6)
        num_episodes_to_plot = min(6, len(self.episode_data))
        
        print(f"Generating combined trajectory plots for {num_episodes_to_plot} episodes...")
        
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
        print(f"Combined trajectory plots saved to: {trajectory_file}")
        
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
        
        print(f"LaTeX table generated: {latex_file}")
        return latex_file
    
    def save_performance_data(self, comment: str = "performance_eval") -> None:
        """
        Save performance data using Logger's save methods plus additional analysis data.
        
        Args:
            comment: Comment to add to saved files
        """
        print("Saving performance data...")
        
        # Use Logger's built-in save methods
        self.save()  # Save as .npy format
        self.save_as_csv(comment)  # Save as CSV files
        
        # Save additional performance metrics as JSON
        performance_file = os.path.join(self.performance_dir, f"performance_metrics_{comment}.json")
        with open(performance_file, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2, cls=NumpyEncoder)
        
        # Save episode data
        episodes_file = os.path.join(self.performance_dir, f"episode_data_{comment}.json")
        with open(episodes_file, 'w') as f:
            json.dump(self.episode_data, f, indent=2, cls=NumpyEncoder)
            
        print(f"Performance data saved:")
        print(f"   Logger data: {self.OUTPUT_FOLDER}")
        print(f"   Performance metrics: {performance_file}")
        print(f"   Episode data: {episodes_file}")
    
    def plot_performance_vs_logger(self, save_plots: bool = True) -> None:
        """
        Generate performance plots using Logger's plot method plus additional analysis.
        
        Args:
            save_plots: Whether to save plots to files
        """
        print("Generating Logger-style plots plus performance analysis...")
        
        # Use Logger's built-in plot method
        self.plot(pwm=False)
        
        # Generate additional performance visualizations
        self.visualize_performance(save_plots)
        
        # Generate individual trajectory plots
        self._generate_individual_trajectory_plots()
        
        print("All plots generated successfully!")
    
    def get_logger_summary(self) -> Dict:
        """
        Get a summary of the Logger data for inspection.
        
        Returns:
            Dictionary with Logger data summary
        """
        summary = {
            'num_drones': self.NUM_DRONES,
            'logging_freq': self.LOGGING_FREQ_HZ,
            'total_episodes': self.current_episode_idx,
            'data_shape': {
                'timestamps': self.timestamps.shape,
                'states': self.states.shape,
                'controls': self.controls.shape
            },
            'counters': self.counters.tolist(),
            'state_labels': [
                'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
                'roll', 'pitch', 'yaw', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
                'rpm0', 'rpm1', 'rpm2', 'rpm3'
            ],
            'control_labels': [
                'pos_x_cmd', 'pos_y_cmd', 'pos_z_cmd', 'vel_x_cmd', 'vel_y_cmd', 'vel_z_cmd',
                'roll_cmd', 'pitch_cmd', 'yaw_cmd', 'ang_vel_x_cmd', 'ang_vel_y_cmd', 'ang_vel_z_cmd'
            ]
        }
        return summary


# Export the main class
__all__ = ['DronePerformanceLogger', 'PerformanceMetrics']
