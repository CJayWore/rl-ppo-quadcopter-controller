#!/usr/bin/env python3
"""
Drone Performance Evaluation Module
"""
import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.signal import find_peaks

# Using matplotlib with a clean, modern style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'gray'
plt.rcParams['grid.color'] = 'lightgray'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7
# Set up matplotlib color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'utils')
sys.path.insert(0, utils_path)
from Logger import Logger

import warnings
warnings.filterwarnings('ignore')


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
    avg_completion_time: float = 0.0  # Average task completion time (scalar)
    control_energy: List[float] = field(default_factory=list)
    safety_events: List[Dict] = field(default_factory=list)
    
    # Medium Priority Metrics  
    attitude_stability: List[float] = field(default_factory=list)
    completion_time_records: List[float] = field(default_factory=list)  # Individual completion times (list)
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
            'avg_completion_time': self.avg_completion_time,  # Average completion time (scalar)
            'control_energy': self.control_energy,
            'safety_events': self.safety_events,
            'attitude_stability': self.attitude_stability,
            'completion_time_records': self.completion_time_records,  # Individual completion times (list)
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
        if hasattr(self, 'completion_time_records_10_percent'):
            result['completion_time_records_10_percent'] = self.completion_time_records_10_percent
            
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
        
        # Initialize tracking for hover-phase position errors
        self.has_reached_target = False
        if hasattr(self, 'current_hover_errors'):
            del self.current_hover_errors
        
        # print(f"Episode {self.current_episode_idx} started for drone {drone_id}")

    def update_target_position(self, info: Dict) -> None:
        """Update target position from environment info."""
        if 'target_pos' in info and not self.target_updated_this_episode:
            self.target_position = np.array(info['target_pos'])
            self.target_updated_this_episode = True
            # print(f"Updated target position: {self.target_position}")

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
        
        # Only record position errors for performance evaluation after reaching target area
        # This avoids including normal navigation approach errors in performance metrics
        if hasattr(self, 'has_reached_target') and self.has_reached_target:
            # Record position error only after first reaching the target
            if not hasattr(self, 'current_hover_errors'):
                self.current_hover_errors = []
            self.current_hover_errors.append(position_error)
        
        # Debug logging removed for production
        
        # Track task completion (sustained hover at target)
        if position_error <= self.target_tolerance:
            if self.hover_start_time is None:
                self.hover_start_time = timestamp
                self.has_reached_target = True  # Mark that we've reached target for the first time
                # print(f"Started hovering at target (error: {position_error:.3f}m)")
            elif timestamp - self.hover_start_time >= self.required_hover_time and not self.task_completed:
                self.task_completed = True
                self.completion_time = timestamp
                print(f"Task completed! Completion time: {self.completion_time:.2f}s")
        else:
            if self.hover_start_time is not None:
                print(f"⚠️  Left target area (error: {position_error:.3f}m)")
                
                # Record safety event when leaving target area (after having reached it)
                if not hasattr(self, 'current_safety_events'):
                    self.current_safety_events = []
                self.current_safety_events.append({
                    'type': 'left_target_area',
                    'timestamp': timestamp,
                    'error': position_error
                })
                
                self.hover_start_time = None
            
            # Check for safety events
            safety_events = []
            
            # 1. Attitude violations (covers most crash scenarios)
            if np.any(np.abs(attitude) > np.pi/3):  # Large attitude angle (60 degrees)
                safety_events.append({
                    'type': 'attitude_violation', 
                    'timestamp': timestamp,
                    'attitude': attitude.copy(),
                    'max_angle': np.max(np.abs(attitude))
                })
            
            # 2. Ground collision detection
            if position[2] < 0.1:  # Very close to ground (assuming ground is at z=0)
                safety_events.append({
                    'type': 'ground_collision',
                    'timestamp': timestamp,
                    'altitude': position[2]
                })
            
            # 3. Extreme velocity detection (potential loss of control)
            velocity_magnitude = np.linalg.norm(velocity)
            if velocity_magnitude > 10.0:  # Very high velocity (m/s)
                safety_events.append({
                    'type': 'excessive_velocity',
                    'timestamp': timestamp,
                    'velocity_magnitude': velocity_magnitude
                })
            
            # 4. Workspace boundary violation
            # Assuming reasonable workspace limits
            workspace_limits = {
                'x_min': -5, 'x_max': 8,
                'y_min': -5, 'y_max': 8, 
                'z_min': 0, 'z_max': 5
            }
            if (position[0] < workspace_limits['x_min'] or position[0] > workspace_limits['x_max'] or
                position[1] < workspace_limits['y_min'] or position[1] > workspace_limits['y_max'] or
                position[2] < workspace_limits['z_min'] or position[2] > workspace_limits['z_max']):
                safety_events.append({
                    'type': 'workspace_violation',
                    'timestamp': timestamp,
                    'position': position.copy()
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
        
        # Calculate all position errors for trajectory visualization
        all_errors = np.array([np.linalg.norm(pos - self.target_position) for pos in positions])
        
        # Use hover-phase errors for performance metrics (only errors after reaching target)
        if hasattr(self, 'current_hover_errors') and len(self.current_hover_errors) > 0:
            hover_errors = np.array(self.current_hover_errors)
            # print(f"   Using {len(hover_errors)} hover-phase errors for performance metrics")
        else:
            # Fallback: if no hover errors recorded, use errors from when drone was close to target
            close_to_target_mask = all_errors <= (self.target_tolerance * 2.0)  # Within 2x tolerance
            if np.any(close_to_target_mask):
                hover_errors = all_errors[close_to_target_mask]
                # print(f"   Fallback: Using {len(hover_errors)} close-to-target errors for performance metrics")
            else:
                # Last resort: use all errors but with a warning
                hover_errors = all_errors
                # print(f"   ⚠️  Warning: Using all {len(hover_errors)} errors (no hover phase detected)")
        
        # High Priority Metrics - use hover-phase errors for accurate performance assessment
        self.metrics.position_errors.extend(hover_errors.tolist())
        
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
        
        # Update overall completion rate and average completion time
        self.metrics.task_completion_rate = np.mean(self.metrics.episode_completion_rates)
        self.metrics.avg_completion_time = np.mean(self.metrics.episode_completion_times)
        
        # Control energy (using RPM data)
        if len(rpms) > 0:
            # Calculate control energy as sum of squared RPM values
            control_energy = np.sum(np.sum(rpms**2, axis=1))
            self.metrics.control_energy.append(control_energy)
            
            # Calculate control smoothness for this episode based on RPM variations
            # Smoothness = inverse of RPM change rate
            if len(rpms) > 1:
                # Calculate RPM differences between consecutive time steps
                rpm_diffs = np.diff(rpms, axis=0)  # Shape: (episode_length-1, 4)
                # Calculate RMS of RPM changes (lower = smoother)
                rms_rpm_change = np.sqrt(np.mean(rpm_diffs**2))
                # Convert to smoothness score (0-1, higher = smoother)
                episode_smoothness = 1.0 / (1.0 + rms_rpm_change / 1000.0)  # Normalize by typical RPM change
                
                # Store episode-level smoothness
                if not hasattr(self.metrics, 'episode_control_smoothness'):
                    self.metrics.episode_control_smoothness = []
                self.metrics.episode_control_smoothness.append(episode_smoothness)
                
                # print(f"   Episode control smoothness: {episode_smoothness:.4f} (RMS change: {rms_rpm_change:.2f})")
            else:
                print("   ⚠️  Not enough RPM data for smoothness calculation")
        else:
            print("   ⚠️  No RPM data available for control metrics")
        
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
        
        # Task completion time: Time from episode start to successful task completion
        # Only recorded for episodes that successfully complete the task (sustained hover)
        if self.task_completed and self.completion_time > 0:
            # Calculate task completion time from episode start to task completion
            individual_completion_time = self.completion_time - timestamps[0]
            self.metrics.completion_time_records.append(individual_completion_time)
            print(f"   Task completion time: {individual_completion_time:.2f}s")
        else:
            # For failed episodes, no task completion time is recorded
            print(f"   Task completion time: N/A (task not completed)")
        
        # Frequency characteristics (dominant frequency of position error)
        if len(hover_errors) > 10:
            try:
                fft_errors = np.fft.fft(hover_errors)
                freqs = np.fft.fftfreq(len(hover_errors), 1/self.LOGGING_FREQ_HZ)
                dominant_freq = freqs[np.argmax(np.abs(fft_errors[1:len(fft_errors)//2])) + 1]
                if 'position_error' not in self.metrics.frequency_characteristics:
                    self.metrics.frequency_characteristics['position_error'] = []
                self.metrics.frequency_characteristics['position_error'].append(abs(dominant_freq))
            except:
                pass  # Skip if FFT fails
        
        # Robustness score (inverse of error variance)
        if len(hover_errors) > 1:
            error_variance = np.var(hover_errors)
            robustness_score = 1.0 / (1.0 + error_variance)
            self.metrics.robustness_scores.append(robustness_score)
        
        # Store episode data
        # Use the last recorded position to avoid contamination from env reset
        actual_final_position = self.last_recorded_position.tolist() if self.last_recorded_position is not None else (positions[-1].tolist() if len(positions) > 0 else [0, 0, 0])
        actual_final_error = self.last_recorded_error if self.last_recorded_error is not None else (all_errors[-1] if len(all_errors) > 0 else float('inf'))
        
        episode_summary = {
            'episode_id': self.current_episode_idx,
            'duration': episode_duration,
            'step_count': episode_length,
            'positions': positions.tolist(),
            'velocities': velocities.tolist(),
            'attitudes': attitudes.tolist(),
            'controls': rpms.tolist(),  # Store RPM as controls for visualization
            'errors': all_errors.tolist(),  # Store all errors for trajectory visualization
            'hover_errors': hover_errors.tolist(),  # Store hover-phase errors for analysis
            'timestamps': timestamps.tolist(),
            'task_completed': self.task_completed,
            'completion_time': completion_time,
            'completion_rate': completion_rate,
            'safety_events_count': safety_events_count,
            'target_position': self.target_position.tolist(),
            'final_position': actual_final_position,
            'final_error': actual_final_error,
            'mean_error': np.mean(hover_errors) if len(hover_errors) > 0 else float('inf'),
            'max_error': np.max(hover_errors) if len(hover_errors) > 0 else float('inf'),
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
        
        # Settling time (simplified implementation) - use completion time records
        if len(self.metrics.completion_time_records) > 0:
            self.metrics.settling_time = np.mean(self.metrics.completion_time_records)
        
        # Control smoothness - Improved version using episode-level smoothness
        # print(f"🔧 Control Smoothness Debug:")
        if hasattr(self.metrics, 'episode_control_smoothness') and len(self.metrics.episode_control_smoothness) > 0:
            # Use the new episode-level smoothness data
            episode_smoothness_values = self.metrics.episode_control_smoothness
            self.metrics.control_smoothness = np.mean(episode_smoothness_values)
            # print(f"   Episode smoothness values: {episode_smoothness_values}")
            # print(f"   Average control_smoothness: {self.metrics.control_smoothness}")
        elif len(self.metrics.control_energy) > 0:
            # Fallback to energy-based calculation
            energy_values = np.array(self.metrics.control_energy)
            # print(f"   Using fallback energy-based calculation")
            # print(f"   Control energy count: {len(self.metrics.control_energy)}")
            # print(f"   Energy values: {energy_values}")
            
            if len(energy_values) > 1:
                # Multiple episodes: use standard deviation for smoothness
                std_energy = np.std(energy_values)
                self.metrics.control_smoothness = 1.0 / (1.0 + std_energy / np.mean(energy_values))  # Normalize by mean
                # print(f"   Energy std: {std_energy}")
                # print(f"   Energy mean: {np.mean(energy_values)}")
                # print(f"   Computed control_smoothness (multi-episode): {self.metrics.control_smoothness}")
            else:
                # Single episode: use a different approach based on energy magnitude
                single_energy = energy_values[0]
                # Normalize based on typical energy ranges (lower energy = smoother control)
                # Assume typical range: 0 to 1e10 (adjust based on your data)
                typical_max_energy = 1e10
                normalized_energy = min(1.0, single_energy / typical_max_energy)
                self.metrics.control_smoothness = 1.0 - normalized_energy  # Invert: lower energy = higher smoothness
                # print(f"   Single energy value: {single_energy}")
                # print(f"   Normalized energy: {normalized_energy}")
                # print(f"   Computed control_smoothness (single-episode): {self.metrics.control_smoothness}")
        else:
            print("   ⚠️  No control data available")
            self.metrics.control_smoothness = 0.0
    
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
            f.write(f"- **Average Task Completion Time**: {self.metrics.avg_completion_time:.2f} seconds\n")
            f.write(f"- **Position RMSE**: {self.metrics.rmse_position:.4f} m\n")
            f.write(f"- **Maximum Position Error**: {self.metrics.max_position_error:.4f} m\n")
            f.write(f"- **Average Control Energy**: {np.mean(self.metrics.control_energy):.2f}\n")
            f.write(f"- **Safety Events**: {len(self.metrics.safety_events)}\n\n")
            
            # Medium Priority Metrics
            f.write("## Medium Priority Metrics\n\n")
            f.write(f"- **Average Attitude Stability**: {np.mean(self.metrics.attitude_stability):.4f}\n")
            if len(self.metrics.completion_time_records) > 0:
                f.write(f"- **Task Completion Time (Successful Episodes)**: {np.mean(self.metrics.completion_time_records):.2f} seconds\n")
            else:
                f.write(f"- **Task Completion Time**: N/A (no successful completions)\n")
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
            errors_array = np.array(self.metrics.position_errors)
            
            # Apply rolling average smoothing as suggested by advisor
            window_size = min(100, len(errors_array) // 20)  # Smooth over 100 points or 5% of data
            if window_size > 1 and len(errors_array) > window_size:
                # Calculate rolling average
                smoothed_errors = np.convolve(errors_array, np.ones(window_size)/window_size, mode='same')
                
                # Downsample for visualization (much less frequent sampling)
                downsample_factor = max(1, len(errors_array) // 1000)  # Target 1000 points max
                steps = np.arange(0, len(errors_array), downsample_factor)
                smoothed_values = smoothed_errors[::downsample_factor]
                
                ax1.plot(steps, smoothed_values, 'b-', alpha=0.8, linewidth=2, label=f'Smoothed (window={window_size})')
                
                # Add confidence interval (optional)
                if len(errors_array) > 200:
                    # Calculate rolling std for confidence bounds
                    rolling_std = np.array([np.std(errors_array[max(0, i-window_size//2):min(len(errors_array), i+window_size//2)]) 
                                          for i in range(len(errors_array))])
                    std_values = rolling_std[::downsample_factor]
                    ax1.fill_between(steps, smoothed_values - std_values, smoothed_values + std_values, 
                                   alpha=0.2, color='blue', label='±1σ')
            else:
                # Fallback for small datasets
                ax1.plot(errors_array, 'b-', alpha=0.7, linewidth=1, label='Position Error')
            
            ax1.set_title('Position Error Time Series (Smoothed)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('Position Error (m)')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
        
        # 2. Position Error Distribution (Histogram)
        ax2 = fig.add_subplot(gs[0, 1])
        if len(self.metrics.position_errors) > 0:
            # Create histogram for better error distribution visualization
            errors_array = np.array(self.metrics.position_errors)
            n_bins = min(50, max(10, len(errors_array) // 20))  # Adaptive bin count
            
            ax2.hist(errors_array, bins=n_bins, alpha=0.7, color='lightblue', 
                    edgecolor='blue', linewidth=0.5, density=True)
            
            # Add mean and median lines
            mean_error = np.mean(errors_array)
            median_error = np.median(errors_array)
            ax2.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.3f}m')
            ax2.axvline(median_error, color='green', linestyle='--', linewidth=2, label=f'Median: {median_error:.3f}m')
            
            ax2.set_title('Position Error Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Position Error (m)')
            ax2.set_ylabel('Frequency')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Add statistics text
            std_error = np.std(errors_array)
            ax2.text(0.02, 0.98, f'Mean: {mean_error:.3f}m\nStd: {std_error:.3f}m\nMedian: {median_error:.3f}m\nn={len(errors_array)}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Control Energy Distribution (Box Plot)
        ax3 = fig.add_subplot(gs[0, 2])
        if len(self.metrics.control_energy) > 0:
            # Box plot for control energy distribution as suggested
            bp = ax3.boxplot([self.metrics.control_energy], patch_artist=True,
                           boxprops=dict(facecolor='lightgreen', alpha=0.7),
                           whiskerprops=dict(color='green', linewidth=1.5),
                           capprops=dict(color='green', linewidth=1.5),
                           medianprops=dict(color='red', linewidth=2),
                           flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
            
            ax3.set_title('Control Energy Distribution', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Control Energy')
            ax3.set_xticklabels(['All Episodes'])
            ax3.grid(True, alpha=0.3)
            
            # Add statistics
            mean_energy = np.mean(self.metrics.control_energy)
            std_energy = np.std(self.metrics.control_energy)
            median_energy = np.median(self.metrics.control_energy)
            ax3.text(0.02, 0.98, f'Mean: {mean_energy:.2e}\nStd: {std_energy:.2e}\nMedian: {median_energy:.2e}\nn={len(self.metrics.control_energy)}', 
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Task Completion Rate Summary (Bar chart only)
        ax4 = fig.add_subplot(gs[1, 0])
        if hasattr(self.metrics, 'episode_completion_rates') and len(self.metrics.episode_completion_rates) > 0:
            episodes = range(1, len(self.metrics.episode_completion_rates) + 1)
            
            # Create bar chart with different colors for success/failure
            colors = ['green' if rate > 0 else 'red' for rate in self.metrics.episode_completion_rates]
            ax4.bar(episodes, self.metrics.episode_completion_rates, color=colors, alpha=0.7)
            
            ax4.set_title('Task Completion Analysis', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Success (1) / Failure (0)')
            ax4.set_ylim(-0.05, 1.05)
            ax4.grid(True, alpha=0.3)
            
            # Add success rate statistics
            success_count = sum(self.metrics.episode_completion_rates)
            total_episodes = len(self.metrics.episode_completion_rates)
            success_rate = (success_count / total_episodes) * 100 if total_episodes > 0 else 0
            
            ax4.text(0.02, 0.98, f'Success Rate: {success_rate:.1f}%', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No Task Completion Data', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
            ax4.set_title('Task Completion Analysis', fontsize=12, fontweight='bold')
        
        # 5. Task Completion Time Analysis
        ax5 = fig.add_subplot(gs[1, 1])
        if len(self.metrics.completion_time_records) > 0:
            ax5.boxplot(self.metrics.completion_time_records, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax5.set_title('Task Completion Time Distribution', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Task Completion Time (s)')
            ax5.grid(True, alpha=0.3)
            
            # Add statistics
            mean_completion = np.mean(self.metrics.completion_time_records)
            std_completion = np.std(self.metrics.completion_time_records)
            ax5.text(0.02, 0.98, f'Mean: {mean_completion:.2f}s\nStd: {std_completion:.2f}s\nSuccessful Episodes: {len(self.metrics.completion_time_records)}', 
                    transform=ax5.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax5.text(0.5, 0.5, 'No Task Completion Time Data\n(No Successful Episodes)', 
                    transform=ax5.transAxes, ha='center', va='center', fontsize=12)
            ax5.set_title('Task Completion Time Distribution', fontsize=12, fontweight='bold')
        
        # 6. Attitude Stability Distribution (Box Plot)
        ax6 = fig.add_subplot(gs[1, 2])
        if len(self.metrics.attitude_stability) > 0:
            # Box plot for attitude stability distribution
            bp = ax6.boxplot([self.metrics.attitude_stability], patch_artist=True,
                           boxprops=dict(facecolor='plum', alpha=0.7),
                           whiskerprops=dict(color='purple', linewidth=1.5),
                           capprops=dict(color='purple', linewidth=1.5),
                           medianprops=dict(color='red', linewidth=2),
                           flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
            
            ax6.set_title('Attitude Stability Distribution', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Stability Score')
            ax6.set_xticklabels(['All Episodes'])
            ax6.grid(True, alpha=0.3)
            
            # Add statistics
            mean_stability = np.mean(self.metrics.attitude_stability)
            std_stability = np.std(self.metrics.attitude_stability)
            ax6.text(0.02, 0.98, f'Mean: {mean_stability:.3f}\nStd: {std_stability:.3f}\nn={len(self.metrics.attitude_stability)}', 
                    transform=ax6.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 7. Robustness Distribution (Box Plot)
        ax7 = fig.add_subplot(gs[2, 0])
        if len(self.metrics.robustness_scores) > 0:
            # Box plot for robustness scores distribution
            bp = ax7.boxplot([self.metrics.robustness_scores], patch_artist=True,
                           boxprops=dict(facecolor='lightcoral', alpha=0.7),
                           whiskerprops=dict(color='darkred', linewidth=1.5),
                           capprops=dict(color='darkred', linewidth=1.5),
                           medianprops=dict(color='red', linewidth=2),
                           flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
            
            ax7.set_title('Robustness Score Distribution', fontsize=12, fontweight='bold')
            ax7.set_ylabel('Robustness Score')
            ax7.set_xticklabels(['All Episodes'])
            ax7.grid(True, alpha=0.3)
            
            # Add statistics
            mean_robustness = np.mean(self.metrics.robustness_scores)
            std_robustness = np.std(self.metrics.robustness_scores)
            ax7.text(0.02, 0.98, f'Mean: {mean_robustness:.3f}\nStd: {std_robustness:.3f}\nn={len(self.metrics.robustness_scores)}', 
                    transform=ax7.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 8. Safety Events Distribution (Box Plot)
        ax8 = fig.add_subplot(gs[2, 1])
        safety_counts = [episode['safety_events_count'] for episode in self.episode_data]
        if safety_counts:
            # Box plot for safety events distribution
            bp = ax8.boxplot([safety_counts], patch_artist=True,
                           boxprops=dict(facecolor='mistyrose', alpha=0.7),
                           whiskerprops=dict(color='red', linewidth=1.5),
                           capprops=dict(color='red', linewidth=1.5),
                           medianprops=dict(color='darkred', linewidth=2),
                           flierprops=dict(marker='o', markerfacecolor='darkred', markersize=4, alpha=0.5))
            
            ax8.set_title('Safety Events Distribution', fontsize=12, fontweight='bold')
            ax8.set_ylabel('Number of Events')
            ax8.set_xticklabels(['All Episodes'])
            ax8.grid(True, alpha=0.3)
            
            # Add statistics
            mean_events = np.mean(safety_counts)
            std_events = np.std(safety_counts)
            total_events = np.sum(safety_counts)
            ax8.text(0.02, 0.98, f'Mean: {mean_events:.1f}\nStd: {std_events:.1f}\nTotal: {total_events}\nn={len(safety_counts)}', 
                    transform=ax8.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 9. Performance Summary Radar Chart
        ax9 = fig.add_subplot(gs[2, 2], projection='polar')
        
        # Prepare data for radar chart
        metrics_names = ['Task Completion', 'Position Accuracy', 'Control Smoothness', 
                        'Attitude Stability', 'Task Completion Speed', 'Robustness']
        
        # Normalize metrics to 0-1 scale
        metrics_values = []
        
        # Task completion rate - this should be the most important metric
        task_completion_rate = self.metrics.task_completion_rate
        metrics_values.append(task_completion_rate)  # Task completion
        print(f"🎯 Radar Chart Task Completion Rate: {task_completion_rate:.4f}")
        
        # Debug: Check the underlying data
        if hasattr(self.metrics, 'episode_completion_rates'):
            print(f"   Episode completion rates: {self.metrics.episode_completion_rates}")
            print(f"   Number of episodes: {len(self.metrics.episode_completion_rates)}")
            print(f"   Successful episodes: {sum(self.metrics.episode_completion_rates)}")
        else:
            print(f"   ⚠️  No episode_completion_rates attribute found!")
        
        # Position accuracy: Use inverse RMSE with better normalization
        if self.metrics.rmse_position > 0:
            # Use 1/(1+RMSE) for better normalization - always between 0 and 1
            metrics_values.append(1.0 / (1.0 + self.metrics.rmse_position))  # Position accuracy
        else:
            metrics_values.append(1.0)
        
        # Control smoothness: Use the computed control_smoothness metric
        if self.metrics.control_smoothness > 0:
            metrics_values.append(min(1.0, self.metrics.control_smoothness))  # Control smoothness
        else:
            # Fallback: use control energy if smoothness not available
            if len(self.metrics.control_energy) > 0:
                # Normalize control efficiency (lower energy is better)
                max_energy = max(self.metrics.control_energy) if self.metrics.control_energy else 1
                avg_energy = np.mean(self.metrics.control_energy)
                # Use inverse normalization: 1/(1+normalized_energy)
                normalized_energy = avg_energy / max_energy
                metrics_values.append(1.0 / (1.0 + normalized_energy))
            else:
                metrics_values.append(0.5)
        
        if len(self.metrics.attitude_stability) > 0:
            avg_stability = np.mean(self.metrics.attitude_stability)
            metrics_values.append(min(1.0, avg_stability * 10))  # Scale up small stability values
        else:
            metrics_values.append(0.5)
        
        # Task Completion Speed: Proper calculation using both completion time and completion rate
        if len(self.metrics.completion_time_records) > 0:
            avg_completion = np.mean(self.metrics.completion_time_records)
            # Task completion speed: shorter time = higher score
            max_reasonable_completion = 10.0  # 10 seconds is reasonable task completion time
            
            # Normalize completion time to speed factor (0-1 scale)
            if avg_completion <= max_reasonable_completion:
                # Good performance: linear normalization
                time_speed_factor = max(0, 1 - avg_completion / max_reasonable_completion)
            else:
                # Poor performance: exponential decay for very slow completion
                time_speed_factor = max(0.05, 1.0 / (avg_completion / max_reasonable_completion))
            
            # CRITICAL FIX: Multiply by completion rate to account for failed episodes
            # This ensures that low completion rates result in lower speeds
            completion_rate = self.metrics.task_completion_rate
            final_speed_score = time_speed_factor * completion_rate
            
            metrics_values.append(final_speed_score)
            print(f"🎯 Task Completion Speed Debug Info (FIXED):")
            print(f"   Average completion time: {avg_completion:.2f}s")
            print(f"   Time speed factor: {time_speed_factor:.4f}")
            print(f"   Task completion rate: {completion_rate:.4f}")
            print(f"   Final speed score: {final_speed_score:.4f}")
        else:
            # No successful task completions
            metrics_values.append(0.0)
            print(f"🎯 Task Completion Speed: 0.0000 (no successful completions)")
        
        if len(self.metrics.robustness_scores) > 0:
            metrics_values.append(np.mean(self.metrics.robustness_scores))
        else:
            metrics_values.append(0.5)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        metrics_values += metrics_values[:1]  # Complete the circle
        
        # Debug: Print radar chart values
        print(f"🎯 Radar Chart Debug Info:")
        for name, value in zip(metrics_names, metrics_values[:-1]):  # Exclude duplicate last value
            print(f"   {name}: {value:.4f}")
        print(f"   RMSE Position: {self.metrics.rmse_position:.4f}")
        print(f"   Control Smoothness: {self.metrics.control_smoothness:.4f}")
        print(f"   Control Energy Count: {len(self.metrics.control_energy)}")
        if len(self.metrics.control_energy) > 0:
            print(f"   Avg Control Energy: {np.mean(self.metrics.control_energy):.2f}")
        
        ax9.plot(angles, metrics_values, 'o-', linewidth=2, color='blue')
        ax9.fill(angles, metrics_values, alpha=0.25, color='blue')
        ax9.set_xticks(angles[:-1])
        ax9.set_xticklabels(metrics_names)
        ax9.set_ylim(0, 1)
        ax9.set_title('Performance Summary', fontsize=12, fontweight='bold', y=1.08)
        ax9.grid(True)
        
        plt.suptitle('Comprehensive Drone Performance Analysis (Improved Visualization)', fontsize=16, fontweight='bold')
        
        if save_plots:
            # Save individual plots only (no combined plot)
            self._save_individual_performance_plots()
            
            # Generate individual trajectory plots
            self._generate_individual_trajectory_plots()
        
        # Removed plt.show() - no longer show the 9-panel combined plot
        plt.close(fig)  # Close the figure to free memory
    
    def _save_individual_performance_plots(self):
        """
        Save each performance plot as an individual image file using improved visualizations.
        """
        print("Saving individual performance plots...")
        
        # Create metrics for radar chart
        metrics_names = ['Task Completion', 'Position Accuracy', 'Control Smoothness', 
                        'Attitude Stability', 'Task Completion Speed', 'Robustness']
        metrics_values = []
        
        # Task completion rate - MUST match the main radar chart calculation
        task_completion_rate = self.metrics.task_completion_rate
        metrics_values.append(task_completion_rate)
        print(f"🎯 Individual Plots Radar Task Completion Rate: {task_completion_rate:.4f}")
        
        # Position accuracy (1 - normalized position error)
        if len(self.metrics.position_errors) > 0:
            avg_position_error = np.mean(self.metrics.position_errors)
            # Normalize to [0,1] - assume max error of 1.0m for normalization
            position_accuracy = max(0, 1 - min(avg_position_error, 1.0))
            metrics_values.append(position_accuracy)
        else:
            metrics_values.append(0.5)
            
        # Control smoothness (already normalized)
        metrics_values.append(self.metrics.control_smoothness)
            
        # Attitude stability (1 - normalized attitude variance)  
        if len(self.metrics.attitude_stability) > 0:
            attitude_score = max(0, 1 - min(np.mean(self.metrics.attitude_stability), 1.0))
            metrics_values.append(attitude_score)
        else:
            metrics_values.append(0.7)  # Default reasonable value
            
        # Task Completion Speed: Proper calculation using both completion time and completion rate
        if len(self.metrics.completion_time_records) > 0:
            avg_completion_time = np.mean(self.metrics.completion_time_records)
            # Normalize task completion time (shorter time = higher speed)
            # Use 10 seconds as reasonable maximum completion time
            max_reasonable_completion = 10.0
            time_speed_factor = max(0, 1 - min(avg_completion_time / max_reasonable_completion, 1.0))
            
            # Consider task completion rate (failed tasks have 0 speed regardless of time)
            completion_rate = self.metrics.task_completion_rate
            final_speed_score = time_speed_factor * completion_rate
            
            metrics_values.append(final_speed_score)
            print(f"🎯 Task Completion Speed (Fixed): {final_speed_score:.4f}")
            print(f"   - Avg completion time: {avg_completion_time:.2f}s")
            print(f"   - Time speed factor: {time_speed_factor:.4f}")  
            print(f"   - Task completion rate: {completion_rate:.4f}")
        else:
            metrics_values.append(0.0)  # No successful completions
            
        # Robustness (average of robustness scores)
        if len(self.metrics.robustness_scores) > 0:
            metrics_values.append(np.mean(self.metrics.robustness_scores))
        else:
            metrics_values.append(0.8)  # Default reasonable value
        
        # Ensure all values are in [0,1] range
        metrics_values = np.clip(metrics_values, 0, 1).tolist()
        metrics_values += metrics_values[:1]  # Complete the circle
        
        # Create angles for radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # 1. Position Error Time Series with Rolling Average and Confidence Intervals
        if len(self.metrics.position_errors) > 0:
            fig1 = plt.figure(figsize=(12, 8))
            ax = fig1.add_subplot(111)
            
            errors_array = np.array(self.metrics.position_errors)
            steps = np.arange(len(errors_array))
            
            # Apply rolling average for smoothing (window size: 100 steps or 5% of data)
            window_size = max(100, int(len(errors_array) * 0.05))
            if len(errors_array) > window_size:
                smoothed_errors = np.convolve(errors_array, np.ones(window_size)/window_size, mode='valid')
                smoothed_steps = steps[window_size//2:len(smoothed_errors)+window_size//2]
                
                # Calculate confidence intervals (assuming normal distribution)
                rolling_std = np.array([np.std(errors_array[max(0, i-window_size//2):min(len(errors_array), i+window_size//2+1)]) 
                                       for i in smoothed_steps])
                upper_bound = smoothed_errors + 1.96 * rolling_std
                lower_bound = smoothed_errors - 1.96 * rolling_std
                
                # Plot confidence interval
                ax.fill_between(smoothed_steps, lower_bound, upper_bound, alpha=0.3, color='lightblue', label='95% Confidence Interval')
                ax.plot(smoothed_steps, smoothed_errors, 'b-', linewidth=2, label=f'Position Error')
                
                # Plot raw data with transparency for context
                if len(errors_array) < 5000:  # Only for smaller datasets
                    ax.plot(steps, errors_array, 'lightgray', alpha=0.5, linewidth=0.5, label='Raw Data')
            else:
                ax.plot(steps, errors_array, 'b-', linewidth=1, label='Position Error')
            
            ax.set_title('Position Error Over Time (Rolling Average, window={window_size})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Position Error (m)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add enhanced statistics
            stats_text = f'Mean: {np.mean(errors_array):.3f}m\nStd: {np.std(errors_array):.3f}m\n'
            stats_text += f'Median: {np.median(errors_array):.3f}m\nMax: {np.max(errors_array):.3f}m\n'
            stats_text += f'95%ile: {np.percentile(errors_array, 95):.3f}m\nData Points: {len(errors_array):,}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            fig1.savefig(os.path.join(self.performance_dir, "01_position_error_time_series.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig1)
        
        # 2. Position Error Distribution (Histogram)
        if len(self.metrics.position_errors) > 0:
            fig2 = plt.figure(figsize=(12, 8))
            ax = fig2.add_subplot(111)
            
            errors_array = np.array(self.metrics.position_errors)
            n_bins = min(50, max(20, len(errors_array) // 30))  # Adaptive bin count for detailed view
            
            # Create histogram with enhanced styling
            n, bins, patches = ax.hist(errors_array, bins=n_bins, alpha=0.7, color='skyblue', 
                                     edgecolor='darkblue', linewidth=0.8, density=True)
            
            # Add statistical overlay lines
            mean_error = np.mean(errors_array)
            median_error = np.median(errors_array)
            std_error = np.std(errors_array)
            
            # Mean and median lines
            ax.axvline(mean_error, color='red', linestyle='--', linewidth=2.5, 
                      label=f'Mean: {mean_error:.3f}m', alpha=0.8)
            ax.axvline(median_error, color='green', linestyle='--', linewidth=2.5, 
                      label=f'Median: {median_error:.3f}m', alpha=0.8)
            
            # Add 95th percentile line
            p95_error = np.percentile(errors_array, 95)
            ax.axvline(p95_error, color='orange', linestyle=':', linewidth=2, 
                      label=f'95th %ile: {p95_error:.3f}m', alpha=0.8)
            
            ax.set_title('Position Error Distribution (Histogram)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Position Error (m)')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Enhanced statistics text
            stats_text = f'Mean: {mean_error:.3f}m\nStd: {std_error:.3f}m\nMedian: {median_error:.3f}m\n'
            stats_text += f'95th %ile: {p95_error:.3f}m\nMax: {np.max(errors_array):.3f}m\n'
            stats_text += f'Data Points: {len(errors_array):,}\nBins: {n_bins}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            fig2.savefig(os.path.join(self.performance_dir, "02_error_distribution_histogram.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig2)
        
        # 3. Control Energy Distribution (Enhanced Box Plot for Large Samples)
        if len(self.metrics.control_energy) > 0:
            fig3 = plt.figure(figsize=(10, 8))
            energy_array = np.array(self.metrics.control_energy)
            
            # Single enhanced boxplot
            ax = fig3.add_subplot(111)
            
            # Enhanced boxplot with outlier handling for large samples
            if len(energy_array) > 1000:  # For large samples
                # Calculate percentiles for outlier filtering
                q1 = np.percentile(energy_array, 25)
                q3 = np.percentile(energy_array, 75)
                iqr = q3 - q1
                
                # Use stricter outlier definition for visualization (1.5 * IQR -> 3.0 * IQR)
                outlier_factor = 3.0 if len(energy_array) > 5000 else 2.0
                lower_fence = q1 - outlier_factor * iqr
                upper_fence = q3 + outlier_factor * iqr
                
                # Create boxplot with custom whisker range
                box = ax.boxplot([energy_array], patch_artist=True,
                                 boxprops=dict(facecolor='lightgreen', alpha=0.7),
                                 medianprops=dict(color='red', linewidth=2),
                                 whiskerprops=dict(linewidth=2),
                                 capprops=dict(linewidth=2),
                                 whis=(5, 95),  # Use 5th to 95th percentile for whiskers
                                 showfliers=True,  # Still show outliers
                                 flierprops=dict(marker='o', markersize=2, alpha=0.3))
                
                ax.set_title('Control Energy Distribution (Enhanced Boxplot)\nWhiskers: 5th-95th Percentile', 
                            fontsize=14, fontweight='bold')
            else:
                # Standard boxplot for smaller samples
                box = ax.boxplot([energy_array], patch_artist=True,
                                 boxprops=dict(facecolor='lightgreen', alpha=0.7),
                                 medianprops=dict(color='red', linewidth=2),
                                 whiskerprops=dict(linewidth=2),
                                 capprops=dict(linewidth=2))
                
                ax.set_title('Control Energy Distribution (Boxplot)', fontsize=14, fontweight='bold')
            
            ax.set_xticklabels(['Control Energy'])
            ax.set_ylabel('Control Energy')
            ax.grid(True, alpha=0.3)
            
            # Format y-axis in scientific notation if values are large
            if np.max(energy_array) > 1e6:
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            # Enhanced statistics with percentile information
            p5, p25, p50, p75, p95 = np.percentile(energy_array, [5, 25, 50, 75, 95])
            stats_text = f'Mean: {np.mean(energy_array):.2e}\nStd: {np.std(energy_array):.2e}\n'
            stats_text += f'Median: {p50:.2e}\n95%ile: {p95:.2e}\n'
            stats_text += f'IQR: {p75-p25:.2e}\nData Points: {len(energy_array):,}'
            
            # Add outlier information for large samples
            if len(energy_array) > 1000:
                q1, q3 = np.percentile(energy_array, [25, 75])
                iqr = q3 - q1
                outliers = energy_array[(energy_array < q1 - 1.5*iqr) | (energy_array > q3 + 1.5*iqr)]
                stats_text += f'\nOutliers: {len(outliers)} ({len(outliers)/len(energy_array)*100:.1f}%)'
                
            # Add whisker information for large samples
            if len(energy_array) > 1000:
                stats_text += f'\nWhiskers: 5-95% percentiles\nto handle large sample size'
            
            # Position statistics text
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            fig3.savefig(os.path.join(self.performance_dir, "03_control_energy_episodes.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig3)
        
        # 4. Task Completion Analysis (Bar chart only)
        fig4 = plt.figure(figsize=(12, 8))
        ax = fig4.add_subplot(111)
        if hasattr(self.metrics, 'episode_completion_rates') and len(self.metrics.episode_completion_rates) > 0:
            episodes = range(1, len(self.metrics.episode_completion_rates) + 1)
            
            # Create bar chart with different colors for success/failure
            colors = ['green' if rate > 0 else 'red' for rate in self.metrics.episode_completion_rates]
            ax.bar(episodes, self.metrics.episode_completion_rates, color=colors, alpha=0.7)
            
            ax.set_ylim(-0.05, 1.05)
            
            # Add success rate statistics
            success_count = sum(self.metrics.episode_completion_rates)
            total_episodes = len(self.metrics.episode_completion_rates)
            success_rate = (success_count / total_episodes) * 100 if total_episodes > 0 else 0
            
            ax.text(0.02, 0.98, f'Success Rate: {success_rate:.1f}%', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No Task Completion Data', 
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
        
        ax.set_title('Task Completion Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success (1) / Failure (0)')
        ax.grid(True, alpha=0.3)
        fig4.savefig(os.path.join(self.performance_dir, "04_task_completion_analysis.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        # 5. Task Completion Time Distribution (Boxplot Only)
        if len(self.metrics.completion_time_records) > 0:
            fig5 = plt.figure(figsize=(10, 8))
            completion_array = np.array(self.metrics.completion_time_records)
            
            # Single boxplot only
            ax = fig5.add_subplot(111)
            
            box = ax.boxplot([completion_array], patch_artist=True,
                            boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2),
                            whiskerprops=dict(linewidth=2),
                            capprops=dict(linewidth=2))
            
            ax.set_xticklabels(['Task Completion Time'])
            ax.set_title('Task Completion Time Distribution', fontsize=14, fontweight='bold')
            ax.set_ylabel('Task Completion Time (s)')
            ax.grid(True, alpha=0.3)
            
            # Enhanced statistics
            p5, p25, p50, p75, p95 = np.percentile(completion_array, [5, 25, 50, 75, 95])
            stats_text = f'Mean: {np.mean(completion_array):.2f}s\nStd: {np.std(completion_array):.2f}s\n'
            stats_text += f'Median: {p50:.2f}s\n95%ile: {p95:.2f}s\n'
            stats_text += f'IQR: {p75-p25:.2f}s\nSuccessful Episodes: {len(completion_array):,}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        else:
            fig5 = plt.figure(figsize=(10, 6))
            ax = fig5.add_subplot(111)
            ax.text(0.5, 0.5, 'No Task Completion Time Data\n(No Successful Episodes)', 
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('Task Completion Time Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        fig5.savefig(os.path.join(self.performance_dir, "05_task_completion_time_distribution.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig5)
        
        # 6. Attitude Stability Distribution (Boxplot Only)  
        if len(self.metrics.attitude_stability) > 0:
            stability_array = np.array(self.metrics.attitude_stability)
            
            # Single boxplot only
            fig6 = plt.figure(figsize=(10, 8))
            ax = fig6.add_subplot(111)
            
            box = ax.boxplot([stability_array], patch_artist=True,
                            boxprops=dict(facecolor='plum', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2),
                            whiskerprops=dict(linewidth=2),
                            capprops=dict(linewidth=2))
            
            ax.set_xticklabels(['Attitude Stability'])
            ax.set_title('Attitude Stability Distribution', fontsize=14, fontweight='bold')
            ax.set_ylabel('Stability Score')
            ax.grid(True, alpha=0.3)
            
            # Enhanced statistics with percentiles
            p5, p25, p50, p75, p95 = np.percentile(stability_array, [5, 25, 50, 75, 95])
            stats_text = f'Mean: {np.mean(stability_array):.3f}\nStd: {np.std(stability_array):.3f}\n'
            stats_text += f'Median: {p50:.3f}\n95%ile: {p95:.3f}\n'
            stats_text += f'IQR: {p75-p25:.3f}\nData Points: {len(stability_array):,}'
            
            # Add outlier info for large samples
            if len(stability_array) > 300:
                q1, q3 = p25, p75
                iqr = q3 - q1
                outliers = stability_array[(stability_array < q1 - 1.5*iqr) | (stability_array > q3 + 1.5*iqr)]
                stats_text += f'\nOutliers: {len(outliers)} ({len(outliers)/len(stability_array)*100:.1f}%)'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            fig6.savefig(os.path.join(self.performance_dir, "06_attitude_stability.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig6)
        
        # 7. Robustness Scores Distribution (Box Plot)
        if len(self.metrics.robustness_scores) > 0:
            fig7 = plt.figure(figsize=(10, 6))
            ax = fig7.add_subplot(111)
            
            box = ax.boxplot([self.metrics.robustness_scores], patch_artist=True,
                            boxprops=dict(facecolor='lightsalmon', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2),
                            whiskerprops=dict(linewidth=2),
                            capprops=dict(linewidth=2))
            
            ax.set_xticklabels(['Robustness Score'])
            ax.set_title('Robustness Scores Distribution', fontsize=14, fontweight='bold')
            ax.set_ylabel('Robustness Score')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            robustness_array = np.array(self.metrics.robustness_scores)
            stats_text = f'Mean: {np.mean(robustness_array):.3f}\nStd: {np.std(robustness_array):.3f}\n'
            stats_text += f'Median: {np.median(robustness_array):.3f}\nData Points: {len(robustness_array):,}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            fig7.savefig(os.path.join(self.performance_dir, "07_robustness_scores.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig7)
        
        # 8. Safety Events Distribution (Boxplot Only)
        safety_counts = [episode['safety_events_count'] for episode in self.episode_data]
        if safety_counts:
            safety_array = np.array(safety_counts)
            
            # Single boxplot for all sample sizes
            fig8 = plt.figure(figsize=(10, 8))
            ax = fig8.add_subplot(111)
            
            # Enhanced boxplot
            box = ax.boxplot([safety_array], patch_artist=True,
                            boxprops=dict(facecolor='lightpink', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2),
                            whiskerprops=dict(linewidth=2),
                            capprops=dict(linewidth=2),
                            whis=(5, 95),  # 5th to 95th percentile whiskers
                            showfliers=True,
                            flierprops=dict(marker='o', markersize=3, alpha=0.6))
            
            ax.set_xticklabels(['Safety Events'])
            ax.set_title('Safety Events Distribution (Boxplot)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Safety Events')
            ax.grid(True, alpha=0.3)
            
            # Enhanced statistics
            p5, p25, p50, p75, p95 = np.percentile(safety_array, [5, 25, 50, 75, 95])
            zero_events = np.sum(safety_array == 0)
            total_events = np.sum(safety_array)
            
            stats_text = f'Mean: {np.mean(safety_array):.1f}\nStd: {np.std(safety_array):.1f}\n'
            stats_text += f'Median: {p50:.1f}\nMax: {np.max(safety_array)}\n'
            stats_text += f'Zero Events: {zero_events}/{len(safety_array)} ({zero_events/len(safety_array)*100:.1f}%)\n'
            stats_text += f'Total Events: {total_events}\nData Points: {len(safety_array):,}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        else:
            fig8 = plt.figure(figsize=(10, 6))
            ax = fig8.add_subplot(111)
            ax.text(0.5, 0.5, 'No Safety Events Data', 
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('Safety Events Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        fig8.savefig(os.path.join(self.performance_dir, "08_safety_events_summary.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig8)
        
        # 9. Performance Summary Radar Chart
        fig9 = plt.figure(figsize=(10, 10))
        ax = fig9.add_subplot(111, projection='polar')
        ax.plot(angles, metrics_values, 'o-', linewidth=3, markersize=8, color='blue')
        ax.fill(angles, metrics_values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Summary (Statistical Visualization Approach)', 
                    fontsize=14, fontweight='bold', y=1.08)
        ax.grid(True)
        
        # Add metric values as text
        for angle, value, name in zip(angles[:-1], metrics_values[:-1], metrics_names):
            ax.text(angle, value + 0.1, f'{value:.2f}', ha='center', va='center', fontweight='bold')
        
        fig9.savefig(os.path.join(self.performance_dir, "09_performance_summary_radar.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig9)
        
        print(f"💾 Saved 9 performance plots to: {self.performance_dir}")
        print("   Files: 01_position_error_time_series.png → 09_performance_summary_radar.png")
    
    def _generate_individual_trajectory_plots(self):
        """
        Generate a single figure with 9 subplot panels showing individual 3D trajectory plots for each episode.
        """
        if not self.episode_data or len(self.episode_data) == 0:
            print("No episode data available for individual trajectory plots")
            return
        
        # Define colors for different episodes
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'red', 'cyan', 'magenta']
        
        # Generate plots for first 9 episodes (or all available if less than 9)
        num_episodes_to_plot = min(9, len(self.episode_data))
        
        print(f"Generating combined trajectory plots for {num_episodes_to_plot} episodes...")
        
        # Create a figure with 3 rows and 3 columns of subplots
        fig = plt.figure(figsize=(18, 18))
        
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
                
            # Create subplot (3 rows, 3 columns)
            ax = fig.add_subplot(3, 3, episode_idx + 1, projection='3d')
            
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
        print(f"Combined 3×3 trajectory plots saved to: {trajectory_file}")
        
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
            f.write(f"Task Completion Time & {self.metrics.avg_completion_time:.2f} s \\\\\n")
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
            if len(self.metrics.completion_time_records) > 0:
                f.write(f"Average Task Completion Time & {np.mean(self.metrics.completion_time_records):.2f} s \\\\\n")
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
