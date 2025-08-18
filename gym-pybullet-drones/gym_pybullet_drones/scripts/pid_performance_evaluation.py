#!/usr/bin/env python3
"""
PID Controller Performance Evaluation Module

This module provides comprehensive performance evaluation capabilities for PID controllers,
including data collection, analysis, and visualization of key performance metrics.
Designed to align with PPO performance evaluation for fair algorithm comparison.
"""

import os
import json
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils'))
    from Logger import Logger
except ImportError:
    print("Warning: Could not import Logger class")

try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except ImportError:
    print("Seaborn not available, using default matplotlib styling")
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
class PIDPerformanceMetrics:
    """Data class to store PID controller performance metrics aligned with PPO version."""
    
    # Core metrics matching PPO version
    position_errors: List[float] = field(default_factory=list)
    task_completion_rate: float = 0.0
    task_completion_time: float = 0.0
    control_energy: List[float] = field(default_factory=list)
    safety_events: List[Dict] = field(default_factory=list)
    
    # Secondary metrics  
    attitude_stability: List[float] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    frequency_characteristics: Dict[str, List[float]] = field(default_factory=dict)
    robustness_scores: List[float] = field(default_factory=list)
    
    # PID-specific metrics
    settling_times: List[float] = field(default_factory=list)
    overshoots: List[float] = field(default_factory=list)
    steady_state_errors: List[float] = field(default_factory=list)
    rise_times: List[float] = field(default_factory=list)
    
    # Episode statistics
    episode_durations: List[float] = field(default_factory=list)
    termination_reasons: List[str] = field(default_factory=list)
    
    # Summary metrics matching PPO version
    rmse_position: float = 0.0
    max_position_error: float = 0.0
    steady_state_error: float = 0.0
    settling_time: float = 0.0
    overshoot: float = 0.0
    control_smoothness: List[float] = field(default_factory=list)
    
    # PID-specific summary metrics
    mae_position: float = 0.0
    average_settling_time: float = 0.0
    average_overshoot: float = 0.0
    control_efficiency: float = 0.0
    hover_success_rate: float = 0.0
    average_hover_time: float = 0.0
    crash_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for JSON serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                result[key] = float(value)
            else:
                result[key] = value
        return result

class PIDPerformanceEvaluator:
    """
    Comprehensive performance evaluator for PID controllers - aligned with PPO version.
    
    This class provides functionality to:
    1. Collect performance data from PID controller simulations
    2. Compute key PID control metrics
    3. Generate detailed visualizations and reports
    4. Save results with _PID suffix for easy identification
    """
    
    def __init__(self, output_folder: str = "results/performance_analysis"):
        """
        Initialize the PID performance evaluator.
        
        Args:
            output_folder: Directory to save results (will add _PID suffix to files)
        """
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        self.metrics = PIDPerformanceMetrics()
        self.episode_data = []
        self.current_episode = {}
        
        # Target position for task completion evaluation (matching PPO version)
        self.target_position = np.array([3, 3, 1.5])  # Default target matching DRLAviary
        self.target_tolerance = 0.15  # meters
        self.required_hover_time = 3.0  # seconds
        
        # Performance tracking
        self.target_positions = []
        self.start_time = None
        self.episode_count = 0
        
        # Episode tracking variables (matching PPO version)
        self.episode_start_time = None
        self.task_completed = False
        self.hover_start_time = None
        self.completion_time = 0.0
        
        print("PID Performance Evaluator initialized")
        print(f"Output folder: {output_folder}")
        print(f"Target position: {self.target_position}")
        print(f"Target tolerance: {self.target_tolerance}m")
    
    def start_evaluation(self, target_pos: np.ndarray = None) -> None:
        """Start a new evaluation session."""
        self.start_time = time.time()
        if target_pos is not None:
            self.target_position = target_pos
        self.target_positions = [self.target_position]
        self.episode_count = 0
        self.episode_data = []
        print("Starting PID performance evaluation")
        print(f"Target positions: {self.target_positions}")
    
    def start_episode(self) -> None:
        """Start tracking a new episode."""
        self.episode_start_time = time.time()
        self.task_completed = False
        self.hover_start_time = None
        self.completion_time = 0.0
        self.current_episode = {
            'start_time': time.time(),
            'positions': [],
            'velocities': [],
            'orientations': [],
            'motor_commands': [],
            'timestamps': [],
            'hover_time': 0.0,
            'termination_reason': 'timeout',
            'task_completed': False
        }
        self.episode_count += 1
        print(f"Episode {self.episode_count} started")
    
    def log_step(self, timestamp: float, state: np.ndarray, target_pos: np.ndarray, 
                 motor_commands: np.ndarray, hover_duration: float) -> None:
        """Log a single simulation step."""
        if not self.current_episode:
            return
        
        # Update target position if provided
        if target_pos is not None:
            self.target_position = target_pos
        
        # Extract state information
        pos = state[0:3]
        quat = state[3:7]
        vel = state[10:13]
        ang_vel = state[13:16]
        
        # Store step data
        self.current_episode['positions'].append(pos.copy())
        self.current_episode['velocities'].append(vel.copy())
        self.current_episode['orientations'].append(quat.copy())
        self.current_episode['motor_commands'].append(motor_commands.copy())
        self.current_episode['timestamps'].append(timestamp)
        self.current_episode['hover_time'] = hover_duration
        
        # Store target position for this episode
        self.current_episode['target_pos'] = self.target_position.copy()
        
        # Compute instantaneous metrics
        position_error = np.linalg.norm(pos - self.target_position)
        self.metrics.position_errors.append(position_error)
        
        # Task completion tracking (matching PPO version)
        within_tolerance = position_error <= self.target_tolerance
        
        if within_tolerance and not self.task_completed:
            if self.hover_start_time is None:
                self.hover_start_time = timestamp
            elif (timestamp - self.hover_start_time) >= self.required_hover_time:
                self.task_completed = True
                self.completion_time = timestamp - self.current_episode['start_time']
                self.current_episode['task_completed'] = True
                print(f"   Task completed at {timestamp:.2f}s (hover time: {self.completion_time:.2f}s)")
        elif not within_tolerance:
            # Reset hover timer if we move out of tolerance
            self.hover_start_time = None
    
    def end_episode(self, termination_reason: str, hover_success: bool = None) -> None:
        """End current episode and compute episode metrics."""
        if not self.current_episode:
            return
        
        # Use task completion status if hover_success not explicitly provided
        if hover_success is None:
            hover_success = self.current_episode.get('task_completed', self.task_completed)
        
        # Store termination info
        self.current_episode['termination_reason'] = termination_reason
        self.current_episode['hover_success'] = hover_success
        self.current_episode['duration'] = time.time() - self.current_episode['start_time']
        
        # Compute episode metrics
        self._compute_episode_metrics()
        
        # Store episode data
        self.episode_data.append(self.current_episode.copy())
        self.current_episode = {}
        
        print(f"Episode {self.episode_count} completed: {termination_reason}")
    
    def _compute_episode_metrics(self) -> None:
        """Compute performance metrics for the current episode - aligned with PPO version."""
        if not self.current_episode['positions']:
            return
        
        positions = np.array(self.current_episode['positions'])
        timestamps = np.array(self.current_episode['timestamps'])
        motor_commands = np.array(self.current_episode['motor_commands'])
        
        # Target position (use current target)
        target_pos = self.target_position
        
        # Position errors over time
        position_errors = [np.linalg.norm(pos - target_pos) for pos in positions]
        
        # Control energy (using motor commands)
        if len(motor_commands) > 0:
            control_energy = np.sum(np.sum(motor_commands**2, axis=1))
            self.metrics.control_energy.append(control_energy)
        
        # Control smoothness (derivative of motor commands)
        if len(motor_commands) > 1:
            motor_derivatives = np.diff(motor_commands, axis=0)
            smoothness = 1.0 / (1.0 + np.mean(np.sum(motor_derivatives**2, axis=1)))
            self.metrics.control_smoothness.append(smoothness)
        
        # Attitude stability (variance in orientation)
        if len(self.current_episode['orientations']) > 1:
            orientations = np.array(self.current_episode['orientations'])
            # Convert quaternions to euler angles for stability measure
            roll_pitch_yaw = []
            for quat in orientations:
                # Simple conversion for roll/pitch stability
                roll = np.arctan2(2*(quat[0]*quat[1] + quat[2]*quat[3]), 
                                1-2*(quat[1]**2 + quat[2]**2))
                pitch = np.arcsin(2*(quat[0]*quat[2] - quat[3]*quat[1]))
                roll_pitch_yaw.append([roll, pitch])
            
            stability = 1.0 / (1.0 + np.std(roll_pitch_yaw))
            self.metrics.attitude_stability.append(stability)
        
        # Response time (time to reach within tolerance for first time)
        first_within_tolerance = np.where(np.array(position_errors) <= self.target_tolerance)[0]
        if len(first_within_tolerance) > 0:
            response_time = timestamps[first_within_tolerance[0]] - timestamps[0]
            self.metrics.response_times.append(response_time)
        
        # Robustness score (inverse of error variance)
        if len(position_errors) > 1:
            error_variance = np.var(position_errors)
            robustness = 1.0 / (1.0 + error_variance)
            self.metrics.robustness_scores.append(robustness)
        
        # Episode duration
        self.metrics.episode_durations.append(self.current_episode['duration'])
        
        # Termination reason
        self.metrics.termination_reasons.append(self.current_episode['termination_reason'])
        
        # PID-specific metrics
        # Settling time (time to reach and stay within 5% of target)
        settling_threshold = self.target_tolerance  # Use target tolerance
        settled_indices = np.where(np.array(position_errors) < settling_threshold)[0]
        if len(settled_indices) > 0:
            # Check for sustained settling (at least 1 second)
            settling_duration = 1.0  # seconds
            min_settle_steps = int(settling_duration * 48)  # assuming 48Hz control
            
            for i in settled_indices:
                if i + min_settle_steps < len(position_errors):
                    # Check if error stays low for required duration
                    sustained = all(np.array(position_errors[i:i+min_settle_steps]) < settling_threshold)
                    if sustained:
                        settling_time = timestamps[i] - timestamps[0]
                        self.metrics.settling_times.append(settling_time)
                        break
            else:
                self.metrics.settling_times.append(timestamps[-1])  # Never settled
        else:
            self.metrics.settling_times.append(timestamps[-1])  # Never settled
        
        # Overshoot (maximum deviation beyond target)
        if len(position_errors) > 0:
            max_error = max(position_errors)
            initial_distance = np.linalg.norm(positions[0] - target_pos)
            if initial_distance > 0:
                overshoot = max(0, (max_error - initial_distance) / initial_distance * 100)
            else:
                overshoot = 0
            self.metrics.overshoots.append(overshoot)
        
        # Steady state error (average error in last 20% of episode)
        if len(position_errors) > 10:
            steady_start = int(0.8 * len(position_errors))
            steady_error = np.mean(position_errors[steady_start:])
            self.metrics.steady_state_errors.append(steady_error)
        
        # Safety events (placeholder - could be enhanced with actual safety checks)
        safety_events_count = 0  # Implement actual safety event detection if needed
        self.current_episode['safety_events_count'] = safety_events_count
    
    def finalize_evaluation(self) -> Dict:
        """Finalize evaluation and compute summary statistics."""
        if len(self.episode_data) == 0:
            print("❌ No episode data available for evaluation")
            return self.metrics.to_dict()
        
        # Compute summary metrics
        self._compute_summary_metrics()
        
        # Generate performance report
        report = self.generate_performance_report()
        
        # Save results
        self.save_performance_data()
        
        # Generate visualizations
        self.visualize_performance()
        
        evaluation_time = time.time() - self.start_time if self.start_time else 0
        print(f"PID Performance evaluation completed in {evaluation_time:.2f}s")
        print(f"Evaluated {len(self.episode_data)} episodes")
        
        return self.metrics.to_dict()
    
    def _compute_summary_metrics(self) -> None:
        """Compute summary statistics from all episodes - aligned with PPO version."""
        if not self.metrics.position_errors:
            print("⚠️  No position error data available for summary computation")
            return
        
        # Position error statistics
        self.metrics.rmse_position = np.sqrt(np.mean(np.array(self.metrics.position_errors)**2))
        self.metrics.mae_position = np.mean(np.abs(self.metrics.position_errors))
        self.metrics.max_position_error = np.max(self.metrics.position_errors)
        
        # Control performance
        if self.metrics.settling_times:
            self.metrics.average_settling_time = np.mean(self.metrics.settling_times)
        if self.metrics.overshoots:
            self.metrics.average_overshoot = np.mean(self.metrics.overshoots)
        
        # Task success rates
        total_episodes = len(self.episode_data)
        if total_episodes > 0:
            successful_episodes = sum(1 for ep in self.episode_data 
                                    if ep.get('task_completed', False))
            hover_successful = sum(1 for ep in self.episode_data if ep.get('hover_success', False))
            crashed_episodes = sum(1 for ep in self.episode_data 
                                 if ep['termination_reason'] == 'crash')
            
            self.metrics.task_completion_rate = successful_episodes / total_episodes * 100
            self.metrics.hover_success_rate = hover_successful / total_episodes * 100
            self.metrics.crash_rate = crashed_episodes / total_episodes * 100
        
            # Average hover time (only for successful episodes)
            if hover_successful > 0:
                successful_hover_times = [ep.get('hover_time', 0) for ep in self.episode_data 
                                        if ep.get('hover_success', False)]
                self.metrics.average_hover_time = np.mean(successful_hover_times)
        
        # Control efficiency (matching PPO version calculation)
        if self.metrics.control_energy and self.metrics.position_errors:
            avg_energy = np.mean(self.metrics.control_energy)
            avg_error = np.mean(self.metrics.position_errors)
            # Efficiency: lower energy with lower error is better
            self.metrics.control_efficiency = 1.0 / (1.0 + avg_energy * avg_error)
        else:
            self.metrics.control_efficiency = 0.0
        
        # Control smoothness summary (compute mean for final reporting)
        if self.metrics.control_smoothness:
            # Store the mean smoothness for reporting while keeping the list
            self.control_smoothness_mean = np.mean(self.metrics.control_smoothness)
        else:
            self.control_smoothness_mean = 0.0
        
        # Print summary of collected metrics for debugging
        print("Final metrics summary:")
        print(f"   Position errors: {len(self.metrics.position_errors)} values")
        print(f"   Control energy: {len(self.metrics.control_energy)} values")
        print(f"   RMSE Position: {self.metrics.rmse_position:.4f}m")
        print(f"   Task completion rate: {self.metrics.task_completion_rate:.1f}%")
        print(f"   Control efficiency: {self.metrics.control_efficiency:.4f}")
    
    def visualize_performance(self, save_plots: bool = True) -> None:
        """
        Generate comprehensive performance visualizations - aligned with PPO version.
        
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
        # Create completion data from episode_data
        completion_rates = [ep.get('task_completed', False) for ep in self.episode_data] if self.episode_data else []
        if completion_rates:
            episodes = range(1, len(completion_rates) + 1)
            ax4.bar(episodes, [1 if x else 0 for x in completion_rates], alpha=0.7, color='green')
            overall_rate = sum(completion_rates) / len(completion_rates) * 100
            ax4.text(0.02, 0.98, f'Overall Rate: {overall_rate:.1f}%', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No Task Completion Data', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.set_title('Task Completion per Episode', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Completed (1=Yes, 0=No)')
        ax4.set_ylim(-0.1, 1.1)
        ax4.grid(True, alpha=0.3)
        
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
        safety_counts = [episode.get('safety_events_count', 0) for episode in self.episode_data]
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
        metrics_names = ['Task Completion', 'Position Accuracy', 'Control Smoothness', 
                        'Attitude Stability', 'Response Speed', 'Robustness']
        
        # Normalize metrics to 0-1 scale
        metrics_values = []
        metrics_values.append(self.metrics.task_completion_rate / 100.0)  # Task completion
        
        # Position accuracy: Use inverse RMSE with better normalization
        if self.metrics.rmse_position > 0:
            # Use 1/(1+RMSE) for better normalization - always between 0 and 1
            metrics_values.append(1.0 / (1.0 + self.metrics.rmse_position))  # Position accuracy
        else:
            metrics_values.append(1.0)
        
        # Control smoothness: Use the computed control_smoothness metric
        if hasattr(self, 'control_smoothness_mean') and self.control_smoothness_mean > 0:
            metrics_values.append(min(1.0, self.control_smoothness_mean))  # Control smoothness
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
        
        # Debug: Print radar chart values
        print("Radar Chart Debug Info:")
        for name, value in zip(metrics_names, metrics_values[:-1]):  # Exclude duplicate last value
            print(f"   {name}: {value:.4f}")
        print(f"   RMSE Position: {self.metrics.rmse_position:.4f}")
        if hasattr(self, 'control_smoothness_mean'):
            print(f"   Control Smoothness (mean): {self.control_smoothness_mean:.4f}")
        else:
            print(f"   Control Smoothness: No data")
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
        
        plt.suptitle('Comprehensive PID Performance Analysis', fontsize=16, fontweight='bold')
        
        if save_plots:
            # Save individual plots only (no combined plot)
            self._save_individual_performance_plots(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, 
                                                  metrics_names, metrics_values, angles)
            
            # Generate individual trajectory plots
            self._generate_individual_trajectory_plots()
        
        plt.show()
    
    def _save_individual_performance_plots(self, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, 
                                         metrics_names, metrics_values, angles):
        """
        Save each performance plot as an individual image file.
        """
        print("Saving individual performance plots...")
        
        # 1. Position Error Time Series
        if len(self.metrics.position_errors) > 0:
            fig1 = plt.figure(figsize=(10, 6))
            ax = fig1.add_subplot(111)
            ax.plot(self.metrics.position_errors, 'b-', alpha=0.7, linewidth=1)
            ax.axhline(y=self.target_tolerance, color='r', linestyle='--', label='Target Tolerance')
            ax.set_title('PID Position Error Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Position Error (m)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig1.savefig(os.path.join(self.output_folder, "01_position_error_time_series_PID.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig1)
        
        # 2. Error Distribution Histogram
        if len(self.metrics.position_errors) > 0:
            fig2 = plt.figure(figsize=(10, 6))
            ax = fig2.add_subplot(111)
            ax.hist(self.metrics.position_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(x=np.mean(self.metrics.position_errors), color='r', linestyle='--', label='Mean')
            ax.set_title('PID Position Error Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Position Error (m)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig2.savefig(os.path.join(self.output_folder, "02_error_distribution_histogram_PID.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig2)
        
        # 3. Control Energy Over Episodes
        if len(self.metrics.control_energy) > 0:
            fig3 = plt.figure(figsize=(10, 6))
            ax = fig3.add_subplot(111)
            episodes = range(1, len(self.metrics.control_energy) + 1)
            ax.plot(episodes, self.metrics.control_energy, 'go-', markersize=6)
            ax.set_title('PID Control Energy per Episode', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Control Energy')
            ax.grid(True, alpha=0.3)
            fig3.savefig(os.path.join(self.output_folder, "03_control_energy_episodes_PID.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig3)
        
        # 4. Task Completion Analysis
        fig4 = plt.figure(figsize=(10, 6))
        ax = fig4.add_subplot(111)
        completion_rates = [ep.get('task_completed', False) for ep in self.episode_data] if self.episode_data else []
        if completion_rates:
            episodes = range(1, len(completion_rates) + 1)
            ax.bar(episodes, [1 if x else 0 for x in completion_rates], alpha=0.7, color='green')
            ax.set_ylim(-0.1, 1.1)
            overall_rate = sum(completion_rates) / len(completion_rates) * 100
            ax.text(0.02, 0.98, f'Overall Rate: {overall_rate:.1f}%', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No Task Completion Data', 
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('PID Task Completion per Episode', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Completed (1=Yes, 0=No)')
        ax.grid(True, alpha=0.3)
        fig4.savefig(os.path.join(self.output_folder, "04_task_completion_analysis_PID.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        # 5. Response Time Analysis
        fig5 = plt.figure(figsize=(10, 6))
        ax = fig5.add_subplot(111)
        if len(self.metrics.response_times) > 0:
            ax.boxplot(self.metrics.response_times, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
            mean_response = np.mean(self.metrics.response_times)
            std_response = np.std(self.metrics.response_times)
            ax.text(0.02, 0.98, f'Mean: {mean_response:.2f}s\nStd: {std_response:.2f}s\nSamples: {len(self.metrics.response_times)}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No Response Time Data', 
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('PID Response Time Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Response Time (s)')
        ax.grid(True, alpha=0.3)
        fig5.savefig(os.path.join(self.output_folder, "05_response_time_distribution_PID.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig5)
        
        # 6. Attitude Stability
        if len(self.metrics.attitude_stability) > 0:
            fig6 = plt.figure(figsize=(10, 6))
            ax = fig6.add_subplot(111)
            episodes = range(1, len(self.metrics.attitude_stability) + 1)
            ax.plot(episodes, self.metrics.attitude_stability, 'mo-', markersize=6)
            ax.set_title('PID Attitude Stability per Episode', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Stability Score')
            ax.grid(True, alpha=0.3)
            fig6.savefig(os.path.join(self.output_folder, "06_attitude_stability_PID.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig6)
        
        # 7. Robustness Scores
        if len(self.metrics.robustness_scores) > 0:
            fig7 = plt.figure(figsize=(10, 6))
            ax = fig7.add_subplot(111)
            episodes = range(1, len(self.metrics.robustness_scores) + 1)
            ax.plot(episodes, self.metrics.robustness_scores, 'co-', markersize=6)
            ax.set_title('PID Robustness Scores per Episode', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Robustness Score')
            ax.grid(True, alpha=0.3)
            fig7.savefig(os.path.join(self.output_folder, "07_robustness_scores_PID.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig7)
        
        # 8. Safety Events Summary
        fig8 = plt.figure(figsize=(10, 6))
        ax = fig8.add_subplot(111)
        safety_counts = [episode.get('safety_events_count', 0) for episode in self.episode_data]
        if safety_counts:
            episodes = range(1, len(safety_counts) + 1)
            ax.bar(episodes, safety_counts, alpha=0.7, color='red')
        ax.set_title('PID Safety Events per Episode', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Number of Events')
        ax.grid(True, alpha=0.3)
        fig8.savefig(os.path.join(self.output_folder, "08_safety_events_summary_PID.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig8)
        
        # 9. Performance Summary Radar Chart
        fig9 = plt.figure(figsize=(10, 10))
        ax = fig9.add_subplot(111, projection='polar')
        ax.plot(angles, metrics_values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, metrics_values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        ax.set_title('PID Performance Summary', fontsize=14, fontweight='bold', y=1.08)
        ax.grid(True)
        fig9.savefig(os.path.join(self.output_folder, "09_performance_summary_radar_PID.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig9)
        
        print(f"Saved 9 individual performance plots to: {self.output_folder}")
        print("   Files: 01_position_error_time_series_PID.png → 09_performance_summary_radar_PID.png")
    
    def _generate_individual_trajectory_plots(self):
        """
        Generate a single figure with 9 subplot panels showing individual 3D trajectory plots for each episode.
        """
        if not self.episode_data or len(self.episode_data) == 0:
            print("❌ No episode data available for trajectory visualization")
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
            episode = self.episode_data[episode_idx]
            positions = episode.get('positions', [])
            if positions:
                all_positions.extend(positions)
            
            # Add target position
            target_pos = episode.get('target_pos', self.target_position)
            all_targets.append(target_pos)
        
        if all_positions:
            all_positions = np.array(all_positions)
            all_targets = np.array(all_targets)
            
            # Combine positions and targets for range calculation
            all_points = np.vstack([all_positions, all_targets])
            
            x_min, x_max = all_points[:, 0].min() - 0.5, all_points[:, 0].max() + 0.5
            y_min, y_max = all_points[:, 1].min() - 0.5, all_points[:, 1].max() + 0.5
            z_min, z_max = all_points[:, 2].min() - 0.5, all_points[:, 2].max() + 0.5
        else:
            # Default ranges
            x_min, x_max = -2, 5
            y_min, y_max = -2, 5
            z_min, z_max = 0, 3
        
        for episode_idx in range(num_episodes_to_plot):
            episode = self.episode_data[episode_idx]
            positions = episode.get('positions', [])
            
            # Create subplot
            ax = fig.add_subplot(3, 3, episode_idx + 1, projection='3d')
            
            if positions and len(positions) > 0:
                positions = np.array(positions)
                
                # Plot trajectory
                color = colors[episode_idx % len(colors)]
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                       c=color, linewidth=2, alpha=0.8, label='Trajectory')
                
                # Mark start and end points
                ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                          c='green', s=100, marker='o', label='Start', alpha=0.9)
                ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                          c='red', s=100, marker='X', label='End', alpha=0.9)
                
                # Plot target position
                target_pos = episode.get('target_pos', self.target_position)
                ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
                          c='gold', s=150, marker='*', label='Target', alpha=0.9)
                
                # Add task completion indicator
                task_completed = episode.get('task_completed', False)
                completion_status = "Completed" if task_completed else "Failed"
                ax.text2D(0.02, 0.98, completion_status, transform=ax.transAxes, 
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.3', 
                                 facecolor='lightgreen' if task_completed else 'lightcoral', 
                                 alpha=0.7))
                
                # Add final error info
                final_error = episode.get('final_error', 
                                        np.linalg.norm(positions[-1] - target_pos) if len(positions) > 0 else float('inf'))
                ax.text2D(0.02, 0.85, f'Final Error: {final_error:.3f}m', 
                         transform=ax.transAxes, fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            else:
                ax.text(0.5, 0.5, 0.5, 'No Position Data', 
                       ha='center', va='center', fontsize=12)
            
            # Set consistent axis limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            
            # Labels and title
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'PID Episode {episode_idx + 1}', fontsize=12, fontweight='bold')
            
            # Add legend only to the first subplot to avoid clutter
            if episode_idx == 0:
                ax.legend(loc='upper right', fontsize=8)
            
            # Set viewing angle for better visualization
            ax.view_init(elev=20, azim=45)
        
        # Add main title
        fig.suptitle('PID Controller: 3D Flight Trajectories (Individual Episodes)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        # Save the combined plot
        trajectory_file = os.path.join(self.output_folder, "trajectories_combined_PID.png")
        plt.savefig(trajectory_file, dpi=300, bbox_inches='tight')
        print(f"Combined 3×3 trajectory plots saved to: {trajectory_file}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        report_lines = [
            "# PID Controller Performance Evaluation Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary Statistics",
            f"- **Episodes Evaluated**: {len(self.episode_data)}",
            f"- **Task Completion Rate**: {self.metrics.task_completion_rate:.1f}%",
            f"- **Hover Success Rate**: {self.metrics.hover_success_rate:.1f}%",
            f"- **Crash Rate**: {self.metrics.crash_rate:.1f}%",
            "",
            "## Position Control Performance",
            f"- **RMSE Position Error**: {self.metrics.rmse_position:.4f} m",
            f"- **Mean Absolute Error**: {self.metrics.mae_position:.4f} m",
            f"- **Maximum Position Error**: {self.metrics.max_position_error:.4f} m",
            f"- **Average Settling Time**: {self.metrics.average_settling_time:.2f} s",
            f"- **Average Overshoot**: {self.metrics.average_overshoot:.1f}%",
            "",
            "## Control Quality",
            f"- **Control Efficiency**: {self.metrics.control_efficiency:.4f}",
            f"- **Average Control Energy**: {np.mean(self.metrics.control_energy) if self.metrics.control_energy else 0:.2f}",
            f"- **Average Control Smoothness**: {getattr(self, 'control_smoothness_mean', 0):.4f}",
            f"- **Average Attitude Stability**: {np.mean(self.metrics.attitude_stability) if self.metrics.attitude_stability else 0:.4f}",
            "",
            "## Episode Statistics",
            f"- **Average Episode Duration**: {np.mean(self.metrics.episode_durations) if self.metrics.episode_durations else 0:.2f} s",
            f"- **Average Hover Time**: {self.metrics.average_hover_time:.2f} s",
            "",
            "## Termination Reasons",
        ]
        
        # Add termination reason statistics
        from collections import Counter
        reason_counts = Counter(self.metrics.termination_reasons)
        for reason, count in reason_counts.items():
            percentage = count / len(self.metrics.termination_reasons) * 100
            report_lines.append(f"- **{reason.title()}**: {count} episodes ({percentage:.1f}%)")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_path = os.path.join(self.output_folder, "performance_report_PID.md")
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"📄 Performance report saved to: {report_path}")
        return report_content
    
    def save_performance_data(self) -> None:
        """Save performance data to JSON file with PID suffix."""
        # Save metrics as JSON
        metrics_path = os.path.join(self.output_folder, "evaluation_results_PID.json")
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_type': 'PID_Controller',
            'episodes_evaluated': len(self.episode_data),
            'metrics': self.metrics.to_dict(),
            'episode_summaries': []
        }
        
        # Add episode summaries
        for i, episode in enumerate(self.episode_data):
            summary = {
                'episode_id': i + 1,
                'duration': episode.get('duration', 0),
                'task_completed': episode.get('task_completed', False),
                'termination_reason': episode.get('termination_reason', 'unknown'),
                'final_error': episode.get('final_error', float('inf')),
                'safety_events_count': episode.get('safety_events_count', 0)
            }
            data['episode_summaries'].append(summary)
        
        with open(metrics_path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        
        print(f"Performance data saved to: {metrics_path}")
        
        # Generate LaTeX table
        self.generate_latex_table()
    
    def generate_latex_table(self) -> str:
        """Generate a LaTeX table of performance metrics."""
        latex_content = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{PID Controller Performance Evaluation Results}}
\\label{{tab:pid_performance}}
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\hline
Episodes Evaluated & {len(self.episode_data)} \\\\
Task Completion Rate & {self.metrics.task_completion_rate:.1f}\\% \\\\
Hover Success Rate & {self.metrics.hover_success_rate:.1f}\\% \\\\
Crash Rate & {self.metrics.crash_rate:.1f}\\% \\\\
\\hline
RMSE Position Error & {self.metrics.rmse_position:.4f} m \\\\
Mean Absolute Error & {self.metrics.mae_position:.4f} m \\\\
Maximum Position Error & {self.metrics.max_position_error:.4f} m \\\\
Average Settling Time & {self.metrics.average_settling_time:.2f} s \\\\
Average Overshoot & {self.metrics.average_overshoot:.1f}\\% \\\\
\\hline
Control Efficiency & {self.metrics.control_efficiency:.4f} \\\\
Average Control Energy & {np.mean(self.metrics.control_energy) if self.metrics.control_energy else 0:.2f} \\\\
Control Smoothness & {getattr(self, 'control_smoothness_mean', 0):.4f} \\\\
Attitude Stability & {np.mean(self.metrics.attitude_stability) if self.metrics.attitude_stability else 0:.4f} \\\\
\\hline
Average Episode Duration & {np.mean(self.metrics.episode_durations) if self.metrics.episode_durations else 0:.2f} s \\\\
Average Hover Time & {self.metrics.average_hover_time:.2f} s \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
        
        # Save LaTeX table
        latex_path = os.path.join(self.output_folder, "performance_table_PID.tex")
        with open(latex_path, 'w') as f:
            f.write(latex_content)
        
        print(f"📝 LaTeX table saved to: {latex_path}")
        return latex_content

# Export the main class
__all__ = ['PIDPerformanceEvaluator', 'PIDPerformanceMetrics']
