#!/usr/bin/env python3
"""
PID Controller Performance Evaluation Module

This module provides comprehensive performance evaluation capabilities for PID controllers,
including data collection, analysis, and visualization of key performance metrics.
Integrates with the existing pid.py script to provide detailed performance analysis.

Author: GitHub Copilot Assistant
Date: August 17, 2025
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

# Import Logger class for data collection
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils'))
    from Logger import Logger
except ImportError:
    print("Warning: Could not import Logger class")

# Try to import seaborn for better styling
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
class PIDPerformanceMetrics:
    """Data class to store PID controller performance metrics."""
    
    # High Priority Metrics (matching PPO version)
    position_errors: List[float] = field(default_factory=list)
    task_completion_rate: float = 0.0
    task_completion_time: float = 0.0  # New metric: time to complete task
    control_energy: List[float] = field(default_factory=list)
    safety_events: List[Dict] = field(default_factory=list)
    
    # Medium Priority Metrics (matching PPO version)  
    attitude_stability: List[float] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    frequency_characteristics: Dict[str, List[float]] = field(default_factory=dict)
    robustness_scores: List[float] = field(default_factory=list)
    
    # PID-specific metrics
    settling_times: List[float] = field(default_factory=list)
    overshoots: List[float] = field(default_factory=list)
    steady_state_errors: List[float] = field(default_factory=list)
    rise_times: List[float] = field(default_factory=list)
    
    # Episode Statistics
    episode_durations: List[float] = field(default_factory=list)
    termination_reasons: List[str] = field(default_factory=list)
    
    # Additional computed metrics (matching PPO version)
    rmse_position: float = 0.0
    max_position_error: float = 0.0
    steady_state_error: float = 0.0
    settling_time: float = 0.0
    overshoot: float = 0.0
    control_smoothness: float = 0.0
    
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
            if isinstance(value, (list, np.ndarray)):
                result[key] = list(value) if len(value) > 0 else []
            else:
                result[key] = float(value) if isinstance(value, (int, float, np.number)) else value
        return result

class PIDPerformanceEvaluator:
    """
    Comprehensive performance evaluator for PID controllers.
    
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
        
        print(f"📊 PID Performance Evaluator initialized")
        print(f"📁 Output folder: {output_folder}")
        print(f"🎯 Target position: {self.target_position}")
        print(f"📏 Target tolerance: {self.target_tolerance}m")
    
    def start_evaluation(self, target_pos: np.ndarray = None) -> None:
        """Start a new evaluation session."""
        self.start_time = time.time()
        if target_pos is not None:
            self.target_position = target_pos
        self.target_positions = [self.target_position]
        self.episode_count = 0
        self.episode_data = []
        print(f"🚀 Starting PID performance evaluation")
        print(f"🎯 Target positions: {self.target_positions}")
    
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
    
    def end_episode(self, termination_reason: str, hover_success: bool) -> None:
        """End current episode and compute episode metrics."""
        if not self.current_episode:
            return
        
        # Store termination info
        self.current_episode['termination_reason'] = termination_reason
        self.current_episode['hover_success'] = hover_success
        self.current_episode['duration'] = time.time() - self.current_episode['start_time']
        
        # Compute episode metrics
        self._compute_episode_metrics()
        
        # Store episode data
        self.episode_data.append(self.current_episode.copy())
        self.current_episode = {}
        
        print(f"📊 Episode {self.episode_count} completed: {termination_reason}")
    
    def _compute_episode_metrics(self) -> None:
        """Compute performance metrics for the current episode."""
        if not self.current_episode['positions']:
            return
        
        positions = np.array(self.current_episode['positions'])
        timestamps = np.array(self.current_episode['timestamps'])
        motor_commands = np.array(self.current_episode['motor_commands'])
        
        # Target position (use first target for simplicity)
        target_pos = self.target_positions[0]
        
        # Position errors over time
        position_errors = [np.linalg.norm(pos - target_pos) for pos in positions]
        
        # Settling time (time to reach and stay within 5% of target)
        settling_threshold = 0.05  # 5cm
        settled_indices = np.where(np.array(position_errors) < settling_threshold)[0]
        if len(settled_indices) > 0:
            # Check for sustained settling (at least 1 second)
            settling_duration = 1.0  # seconds
            min_settle_steps = int(settling_duration * 48)  # assuming 48Hz control
            
            for i in settled_indices:
                if i + min_settle_steps < len(position_errors):
                    if all(np.array(position_errors[i:i+min_settle_steps]) < settling_threshold):
                        settling_time = timestamps[i]
                        self.metrics.settling_times.append(settling_time)
                        break
            else:
                self.metrics.settling_times.append(timestamps[-1])  # Never truly settled
        else:
            self.metrics.settling_times.append(timestamps[-1])  # Never settled
        
        # Overshoot (maximum deviation beyond target)
        if len(position_errors) > 0:
            max_error = max(position_errors)
            initial_distance = np.linalg.norm(positions[0] - target_pos)
            overshoot = max(0, (max_error - initial_distance) / initial_distance * 100)
            self.metrics.overshoots.append(overshoot)
        
        # Steady state error (average error in last 20% of episode)
        if len(position_errors) > 10:
            steady_start = int(0.8 * len(position_errors))
            steady_error = np.mean(position_errors[steady_start:])
            self.metrics.steady_state_errors.append(steady_error)
        
        # Control energy (sum of squared motor commands)
        if len(motor_commands) > 0:
            control_energy = np.sum(np.sum(motor_commands**2, axis=1))
            self.metrics.control_energy.append(control_energy)
        
        # Control smoothness (derivative of motor commands)
        if len(motor_commands) > 1:
            motor_derivatives = np.diff(motor_commands, axis=0)
            smoothness = 1.0 / (1.0 + np.mean(np.sum(motor_derivatives**2, axis=1)))
            self.metrics.control_smoothness.append(smoothness)
            print(f"🔧 Control Smoothness Debug (Episode {len(self.metrics.control_smoothness)}):")
            print(f"   Motor commands shape: {motor_commands.shape}")
            print(f"   Motor derivatives shape: {motor_derivatives.shape}")
            print(f"   Mean derivative magnitude: {np.mean(np.sum(motor_derivatives**2, axis=1)):.4f}")
            print(f"   Smoothness value: {smoothness:.4f}")
        else:
            print(f"⚠️  Control smoothness calculation skipped: insufficient motor commands ({len(motor_commands)})")
        
        # Attitude stability (variance in orientation)
        if len(self.current_episode['orientations']) > 1:
            orientations = np.array(self.current_episode['orientations'])
            # Convert quaternions to euler angles for analysis
            euler_angles = []
            for quat in orientations:
                # Simple quaternion to euler conversion (approximate)
                roll = np.arctan2(2*(quat[3]*quat[0] + quat[1]*quat[2]), 
                                1 - 2*(quat[0]**2 + quat[1]**2))
                pitch = np.arcsin(2*(quat[3]*quat[1] - quat[2]*quat[0]))
                yaw = np.arctan2(2*(quat[3]*quat[2] + quat[0]*quat[1]), 
                               1 - 2*(quat[1]**2 + quat[2]**2))
                euler_angles.append([roll, pitch, yaw])
            
            euler_angles = np.array(euler_angles)
            attitude_variance = np.mean(np.var(euler_angles, axis=0))
            stability = 1.0 / (1.0 + attitude_variance)
            self.metrics.attitude_stability.append(stability)
        
        # Episode duration
        self.metrics.episode_durations.append(self.current_episode['duration'])
        
        # Termination reason
        self.metrics.termination_reasons.append(self.current_episode['termination_reason'])
        
        # Task success metrics
        if self.current_episode['hover_success']:
            self.metrics.average_hover_time += self.current_episode['hover_time']
    
    def finalize_evaluation(self) -> Dict:
        """Finalize evaluation and compute summary statistics."""
        if len(self.episode_data) == 0:
            print("❌ No episode data available for evaluation")
            return {}
        
        # Compute summary metrics
        self._compute_summary_metrics()
        
        # Generate performance report
        report = self.generate_performance_report()
        
        # Save results
        self.save_performance_data()
        
        # Generate visualizations
        self.visualize_performance()
        
        evaluation_time = time.time() - self.start_time if self.start_time else 0
        print(f"✅ PID Performance evaluation completed in {evaluation_time:.2f}s")
        print(f"📊 Evaluated {len(self.episode_data)} episodes")
        
        return self.metrics.to_dict()
    
    def _compute_summary_metrics(self) -> None:
        """Compute summary statistics from all episodes."""
        if not self.metrics.position_errors:
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
        successful_episodes = sum(1 for ep in self.episode_data 
                                if ep['termination_reason'] == 'success')
        hover_successful = sum(1 for ep in self.episode_data if ep.get('hover_success', False))
        crashed_episodes = sum(1 for ep in self.episode_data 
                             if ep['termination_reason'] == 'crash')
        
        self.metrics.task_completion_rate = successful_episodes / total_episodes * 100
        self.metrics.hover_success_rate = hover_successful / total_episodes * 100
        self.metrics.crash_rate = crashed_episodes / total_episodes * 100
        
        # Average hover time (only for successful episodes)
        if hover_successful > 0:
            total_hover_time = sum(ep.get('hover_time', 0) for ep in self.episode_data 
                                 if ep.get('hover_success', False))
            self.metrics.average_hover_time = total_hover_time / hover_successful
        
        # Control efficiency (lower energy with good performance is better)
        if self.metrics.control_energy and self.metrics.position_errors:
            mean_energy = np.mean(self.metrics.control_energy)
            mean_error = np.mean(self.metrics.position_errors)
            # Normalize energy more aggressively for RPM-based control
            # RPM values are typically 10,000-50,000, so RPM^2 can be very large
            normalized_energy = mean_energy / 1e12  # Scale down by 1 trillion
            self.metrics.control_efficiency = 1.0 / (1.0 + normalized_energy * mean_error * 100)
            print(f"🔧 Control Efficiency Debug:")
            print(f"   Mean energy: {mean_energy:.2f}")
            print(f"   Normalized energy: {normalized_energy:.10f}")
            print(f"   Mean error: {mean_error:.4f}")
            print(f"   Control efficiency: {self.metrics.control_efficiency:.4f}")
        else:
            print(f"⚠️  Control efficiency calculation skipped:")
            print(f"   Control energy count: {len(self.metrics.control_energy) if self.metrics.control_energy else 0}")
            print(f"   Position error count: {len(self.metrics.position_errors) if self.metrics.position_errors else 0}")
        
        # Print summary of collected metrics for debugging
        print(f"📊 Final metrics summary:")
        print(f"   Position errors: {len(self.metrics.position_errors)} values")
        print(f"   Control energy: {len(self.metrics.control_energy)} values")
        print(f"   Control smoothness: {len(self.metrics.control_smoothness)} values")
        print(f"   Control efficiency: {self.metrics.control_efficiency:.4f}")
        if self.metrics.control_smoothness:
            print(f"   Avg control smoothness: {np.mean(self.metrics.control_smoothness):.4f}")
        else:
            print(f"   Avg control smoothness: 0.0000 (no data)")
    
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
            f"- **Average Control Smoothness**: {np.mean(self.metrics.control_smoothness) if self.metrics.control_smoothness else 0:.4f}",
            f"- **Average Attitude Stability**: {np.mean(self.metrics.attitude_stability):.4f}",
            "",
            "## Episode Statistics",
            f"- **Average Episode Duration**: {np.mean(self.metrics.episode_durations):.2f} s",
            f"- **Average Hover Time**: {self.metrics.average_hover_time:.2f} s",
            "",
            "## Termination Reasons",
        ]
        
        # Add termination reason statistics
        from collections import Counter
        reason_counts = Counter(self.metrics.termination_reasons)
        for reason, count in reason_counts.items():
            percentage = count / len(self.episode_data) * 100
            report_lines.append(f"- **{reason.title()}**: {count} episodes ({percentage:.1f}%)")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_path = os.path.join(self.output_folder, "performance_report_PID.md")
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"📄 Performance report saved to: {report_path}")
        return report_content
    
    def visualize_performance(self) -> None:
        """Generate comprehensive performance visualizations."""
        if len(self.episode_data) == 0:
            print("❌ No data available for visualization")
            return
        
        # Create comprehensive performance visualization
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Position Error Time Series
        ax1 = fig.add_subplot(gs[0, 0])
        if self.metrics.position_errors:
            ax1.plot(self.metrics.position_errors, 'b-', alpha=0.7, linewidth=1)
            ax1.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5cm threshold')
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('Position Error (m)')
            ax1.set_title('Position Error Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Error Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if self.metrics.position_errors:
            ax2.hist(self.metrics.position_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(x=np.mean(self.metrics.position_errors), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(self.metrics.position_errors):.3f}m')
            ax2.set_xlabel('Position Error (m)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Position Error Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Control Energy per Episode
        ax3 = fig.add_subplot(gs[0, 2])
        if self.metrics.control_energy:
            episodes = range(1, len(self.metrics.control_energy) + 1)
            ax3.plot(episodes, self.metrics.control_energy, 'g-o', markersize=4)
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Control Energy')
            ax3.set_title('Control Energy per Episode')
            ax3.grid(True, alpha=0.3)
        
        # 4. Task Completion Analysis
        ax4 = fig.add_subplot(gs[1, 0])
        success_rate = self.metrics.task_completion_rate
        hover_rate = self.metrics.hover_success_rate
        crash_rate = self.metrics.crash_rate
        timeout_rate = 100 - success_rate - crash_rate
        
        categories = ['Success', 'Hover\nSuccess', 'Crash', 'Timeout']
        values = [success_rate, hover_rate, crash_rate, timeout_rate]
        colors = ['green', 'lightgreen', 'red', 'orange']
        
        bars = ax4.bar(categories, values, color=colors, alpha=0.7)
        ax4.set_ylabel('Percentage (%)')
        ax4.set_title('Task Completion Analysis')
        ax4.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 5. Settling Time Distribution
        ax5 = fig.add_subplot(gs[1, 1])
        if self.metrics.settling_times:
            ax5.hist(self.metrics.settling_times, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax5.axvline(x=np.mean(self.metrics.settling_times), color='r', linestyle='--',
                       label=f'Mean: {np.mean(self.metrics.settling_times):.2f}s')
            ax5.set_xlabel('Settling Time (s)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Settling Time Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Attitude Stability
        ax6 = fig.add_subplot(gs[1, 2])
        if self.metrics.attitude_stability:
            episodes = range(1, len(self.metrics.attitude_stability) + 1)
            ax6.plot(episodes, self.metrics.attitude_stability, 'r-', alpha=0.7)
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Stability Score')
            ax6.set_title('Attitude Stability per Episode')
            ax6.set_ylim(0, 1)
            ax6.grid(True, alpha=0.3)
        
        # 7. Control Smoothness
        ax7 = fig.add_subplot(gs[2, 0])
        if self.metrics.control_smoothness:
            episodes = range(1, len(self.metrics.control_smoothness) + 1)
            ax7.plot(episodes, self.metrics.control_smoothness, 'm-', alpha=0.7)
            ax7.set_xlabel('Episode')
            ax7.set_ylabel('Smoothness Score')
            ax7.set_title('Control Smoothness per Episode')
            ax7.set_ylim(0, 1)
            ax7.grid(True, alpha=0.3)
        
        # 8. Episode Duration Analysis
        ax8 = fig.add_subplot(gs[2, 1])
        if self.metrics.episode_durations:
            ax8.hist(self.metrics.episode_durations, bins=20, alpha=0.7, color='cyan', edgecolor='black')
            ax8.axvline(x=np.mean(self.metrics.episode_durations), color='r', linestyle='--',
                       label=f'Mean: {np.mean(self.metrics.episode_durations):.2f}s')
            ax8.set_xlabel('Episode Duration (s)')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Episode Duration Distribution')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # 9. Performance Summary Radar Chart
        ax9 = fig.add_subplot(gs[2:, 2], projection='polar')
        
        # Prepare radar chart data
        categories = ['Task\nCompletion', 'Hover\nSuccess', 'Position\nAccuracy', 
                     'Control\nEfficiency', 'Attitude\nStability', 'Control\nSmoothness']
        
        # Normalize values to 0-1 scale
        values = [
            self.metrics.task_completion_rate / 100,
            self.metrics.hover_success_rate / 100,
            max(0, 1 - self.metrics.rmse_position),  # Higher accuracy = lower error
            self.metrics.control_efficiency,
            np.mean(self.metrics.attitude_stability) if self.metrics.attitude_stability else 0,
            np.mean(self.metrics.control_smoothness) if self.metrics.control_smoothness else 0
        ]
        
        # Close the radar chart
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax9.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax9.fill(angles, values, alpha=0.25, color='blue')
        ax9.set_xticks(angles[:-1])
        ax9.set_xticklabels(categories)
        ax9.set_ylim(0, 1)
        ax9.set_title('PID Performance Summary\n(Radar Chart)', pad=20)
        ax9.grid(True)
        
        # Add performance metrics as text
        performance_text = (
            f"RMSE: {self.metrics.rmse_position:.3f}m\n"
            f"Success: {self.metrics.task_completion_rate:.1f}%\n"
            f"Settling: {self.metrics.average_settling_time:.2f}s\n"
            f"Efficiency: {self.metrics.control_efficiency:.3f}"
        )
        
        ax9.text(0.02, 0.98, performance_text, transform=ax9.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('PID Controller Performance Analysis', fontsize=16, y=0.95)
        
        # Save the comprehensive plot
        plot_path = os.path.join(self.output_folder, "pid_performance_analysis_PID.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance visualization saved to: {plot_path}")
        
        # Save individual performance plots
        self._save_individual_performance_plots(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, categories, values, angles)
        
        # Generate individual trajectory plots
        self._generate_trajectory_plots()
    
    def _save_individual_performance_plots(self, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, 
                                         categories, values, angles) -> None:
        """Save each performance plot as an individual image file."""
        print("💾 Saving individual performance plots...")
        
        # 1. Position Error Time Series
        if self.metrics.position_errors:
            fig1 = plt.figure(figsize=(12, 6))
            ax = fig1.add_subplot(111)
            ax.plot(self.metrics.position_errors, 'b-', alpha=0.7, linewidth=1.5)
            ax.axhline(y=0.05, color='r', linestyle='--', label='Target Tolerance (5cm)', linewidth=2)
            ax.set_title('PID Position Error Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Position Error (m)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig1.savefig(os.path.join(self.output_folder, "01_position_error_time_series_PID.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig1)
        
        # 2. Error Distribution Histogram
        if self.metrics.position_errors:
            fig2 = plt.figure(figsize=(12, 6))
            ax = fig2.add_subplot(111)
            ax.hist(self.metrics.position_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(x=np.mean(self.metrics.position_errors), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(self.metrics.position_errors):.3f}m', linewidth=2)
            ax.set_title('PID Position Error Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Position Error (m)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig2.savefig(os.path.join(self.output_folder, "02_error_distribution_histogram_PID.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig2)
        
        # 3. Control Energy per Episode
        if self.metrics.control_energy:
            fig3 = plt.figure(figsize=(12, 6))
            ax = fig3.add_subplot(111)
            episodes = range(1, len(self.metrics.control_energy) + 1)
            ax.plot(episodes, self.metrics.control_energy, 'go-', markersize=8, linewidth=2)
            ax.set_title('PID Control Energy per Episode', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Control Energy')
            ax.grid(True, alpha=0.3)
            mean_energy = np.mean(self.metrics.control_energy)
            ax.axhline(y=mean_energy, color='r', linestyle='--', 
                      label=f'Mean: {mean_energy:.2f}', linewidth=2)
            ax.legend()
            fig3.savefig(os.path.join(self.output_folder, "03_control_energy_episodes_PID.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig3)
        
        # 4. Task Completion Analysis
        fig4 = plt.figure(figsize=(12, 6))
        ax = fig4.add_subplot(111)
        success_rate = self.metrics.task_completion_rate
        hover_rate = self.metrics.hover_success_rate
        crash_rate = self.metrics.crash_rate
        timeout_rate = 100 - success_rate - crash_rate
        
        labels = ['Success', 'Hover Success', 'Crash', 'Timeout']
        rates = [success_rate, hover_rate, crash_rate, timeout_rate]
        colors = ['green', 'lightgreen', 'red', 'orange']
        
        bars = ax.bar(labels, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax.set_ylabel('Percentage (%)')
        ax.set_title('PID Task Completion Analysis', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        fig4.savefig(os.path.join(self.output_folder, "04_task_completion_analysis_PID.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        # 5. Settling Time Distribution
        fig5 = plt.figure(figsize=(12, 6))
        ax = fig5.add_subplot(111)
        if self.metrics.settling_times:
            ax.boxplot(self.metrics.settling_times, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
            mean_settling = np.mean(self.metrics.settling_times)
            std_settling = np.std(self.metrics.settling_times)
            ax.text(0.02, 0.98, f'Mean: {mean_settling:.2f}s\nStd: {std_settling:.2f}s\nSamples: {len(self.metrics.settling_times)}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No Settling Time Data Available', 
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('PID Settling Time Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Settling Time (s)')
        ax.grid(True, alpha=0.3)
        fig5.savefig(os.path.join(self.output_folder, "05_settling_time_distribution_PID.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig5)
        
        # 6. Attitude Stability per Episode
        if self.metrics.attitude_stability:
            fig6 = plt.figure(figsize=(12, 6))
            ax = fig6.add_subplot(111)
            episodes = range(1, len(self.metrics.attitude_stability) + 1)
            ax.plot(episodes, self.metrics.attitude_stability, 'mo-', markersize=8, linewidth=2)
            ax.set_title('PID Attitude Stability per Episode', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Stability Score')
            ax.grid(True, alpha=0.3)
            mean_stability = np.mean(self.metrics.attitude_stability)
            ax.axhline(y=mean_stability, color='r', linestyle='--', 
                      label=f'Mean: {mean_stability:.3f}', linewidth=2)
            ax.legend()
            fig6.savefig(os.path.join(self.output_folder, "06_attitude_stability_PID.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig6)
        
        # 7. Control Smoothness per Episode
        if self.metrics.control_smoothness:
            fig7 = plt.figure(figsize=(12, 6))
            ax = fig7.add_subplot(111)
            episodes = range(1, len(self.metrics.control_smoothness) + 1)
            ax.plot(episodes, self.metrics.control_smoothness, 'co-', markersize=8, linewidth=2)
            ax.set_title('PID Control Smoothness per Episode', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Smoothness Score')
            ax.grid(True, alpha=0.3)
            mean_smoothness = np.mean(self.metrics.control_smoothness)
            ax.axhline(y=mean_smoothness, color='r', linestyle='--', 
                      label=f'Mean: {mean_smoothness:.4f}', linewidth=2)
            ax.legend()
            fig7.savefig(os.path.join(self.output_folder, "07_control_smoothness_PID.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig7)
        
        # 8. Episode Duration Analysis
        if self.metrics.episode_durations:
            fig8 = plt.figure(figsize=(12, 6))
            ax = fig8.add_subplot(111)
            episodes = range(1, len(self.metrics.episode_durations) + 1)
            ax.bar(episodes, self.metrics.episode_durations, alpha=0.7, color='purple', 
                   edgecolor='black', linewidth=1)
            ax.set_title('PID Episode Duration Analysis', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Duration (s)')
            ax.grid(True, alpha=0.3, axis='y')
            mean_duration = np.mean(self.metrics.episode_durations)
            ax.axhline(y=mean_duration, color='r', linestyle='--', 
                      label=f'Mean: {mean_duration:.2f}s', linewidth=2)
            ax.legend()
            fig8.savefig(os.path.join(self.output_folder, "08_episode_duration_analysis_PID.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig8)
        
        # 9. Performance Summary Radar Chart
        fig9 = plt.figure(figsize=(10, 10))
        ax = fig9.add_subplot(111, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=3, color='blue', markersize=8)
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('PID Performance Summary\n(Radar Chart)', fontsize=14, fontweight='bold', y=1.08)
        ax.grid(True)
        
        # Add performance metrics as text
        performance_text = (
            f"RMSE: {self.metrics.rmse_position:.3f}m\n"
            f"Success: {self.metrics.task_completion_rate:.1f}%\n"
            f"Settling: {self.metrics.average_settling_time:.2f}s\n"
            f"Efficiency: {self.metrics.control_efficiency:.3f}"
        )
        ax.text(0.02, 0.98, performance_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig9.savefig(os.path.join(self.output_folder, "09_performance_summary_radar_PID.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig9)
        
        print("✅ Saved 9 individual performance plots with _PID suffix:")
        print("   01_position_error_time_series_PID.png → 09_performance_summary_radar_PID.png")
    
    def _generate_trajectory_plots(self) -> None:
        """Generate 3×3 individual trajectory plots similar to performance_evaluation.py."""
        if not self.episode_data or len(self.episode_data) == 0:
            print("📊 No episode data available for trajectory plots")
            return
        
        # Define colors for different episodes
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'red', 'cyan', 'magenta']
        
        # Generate plots for first 9 episodes (or all available if less than 9)
        num_episodes_to_plot = min(9, len(self.episode_data))
        
        print(f"📊 Generating 3×3 trajectory plots for {num_episodes_to_plot} episodes...")
        
        # Create a figure with 3 rows and 3 columns of subplots
        fig = plt.figure(figsize=(20, 20))
        
        # Calculate global min/max ranges for consistent scaling across all subplots
        all_positions = []
        all_targets = []
        
        for episode_idx in range(num_episodes_to_plot):
            episode = self.episode_data[episode_idx]
            if episode['positions']:
                positions = np.array(episode['positions'])
                all_positions.append(positions)
            
            # Use target position from episode data (updated for each episode)
            target_pos = episode.get('target_pos', self.target_positions[0] if self.target_positions else np.array([0, 0, 1]))
            all_targets.append(target_pos.reshape(1, 3))
        
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
            episode = self.episode_data[episode_idx]
            
            if not episode['positions']:
                continue
                
            positions = np.array(episode['positions'])
            # Use episode-specific target position instead of global one
            target_pos = episode.get('target_pos', self.target_positions[0] if self.target_positions else np.array([0, 0, 1]))
            
            # Create subplot (3 rows, 3 columns)
            ax = fig.add_subplot(3, 3, episode_idx + 1, projection='3d')
            
            color = colors[episode_idx % len(colors)]
            
            # Plot trajectory
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   color=color, linewidth=3, label='Trajectory', alpha=0.8)
            
            # Plot start position (first position)
            ax.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 
                      c='green', s=80, marker='o', label='Start', 
                      edgecolors='black', linewidth=2)
            
            # Plot target position (star) - updated for each episode
            ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], 
                      c='red', s=120, marker='*', label='Target', 
                      edgecolors='black', linewidth=2)
            
            # Set consistent ranges for all subplots
            if all_positions:
                ax.set_xlim(global_x_range)
                ax.set_ylim(global_y_range)
                ax.set_zlim(global_z_range)
            
            # Set labels and title
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_zlabel('Z (m)', fontsize=12)
            
            # Episode info in title
            termination = episode.get('termination_reason', 'unknown')
            hover_time = episode.get('hover_time', 0.0)
            duration = episode.get('duration', 0.0)
            
            ax.set_title(f'Episode {episode_idx + 1}\n{termination} | {duration:.1f}s | hover: {hover_time:.1f}s', 
                        fontsize=11, fontweight='bold', pad=15)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend (smaller and positioned better)
            ax.legend(loc='upper right', fontsize=9, markerscale=0.8)
            
            # Adjust tick label size
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
            ax.tick_params(axis='z', labelsize=9)
        
        # Add main title
        fig.suptitle('PID Controller: 3D Flight Trajectories (Individual Episodes)', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        # Save the combined trajectory plot
        trajectory_file = os.path.join(self.output_folder, "trajectories_analysis_PID.png")
        plt.savefig(trajectory_file, dpi=300, bbox_inches='tight')
        print(f"🎯 3×3 trajectory plots saved to: {trajectory_file}")
        
        # Close the figure to free memory
        plt.close(fig)
    
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
                'episode': i + 1,
                'duration': episode['duration'],
                'termination_reason': episode['termination_reason'],
                'hover_success': episode.get('hover_success', False),
                'hover_time': episode.get('hover_time', 0.0),
                'final_position_error': (
                    np.linalg.norm(np.array(episode['positions'][-1]) - self.target_positions[0])
                    if episode['positions'] else 0.0
                )
            }
            data['episode_summaries'].append(summary)
        
        with open(metrics_path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        
        print(f"💾 Performance data saved to: {metrics_path}")
        
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
Control Smoothness & {np.mean(self.metrics.control_smoothness) if self.metrics.control_smoothness else 0:.4f} \\\\
Attitude Stability & {np.mean(self.metrics.attitude_stability):.4f} \\\\
\\hline
Average Episode Duration & {np.mean(self.metrics.episode_durations):.2f} s \\\\
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
