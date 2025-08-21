#!/usr/bin/env python3
"""
PID Parameter Auto-Tuner for drone control systems.

This module provides automated PID parameter tuning using differential evolution
optimization algorithm. It evaluates PID parameters based on stability, tracking
error, overshoot, and energy efficiency metrics for different flight tasks.

Classes:
    PIDTuner: Main tuning class for PID parameter optimization

Functions:
    main: Command-line interface for the tuner
"""

import numpy as np
import json
import os
import sys
import time
import argparse
import shutil
from datetime import datetime
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from pid import run
from gym_pybullet_drones.utils.enums import DroneModel

class PIDTuner:
    """
    PID parameter tuner using differential evolution optimization.
    
    This class automatically searches for optimal PID parameters by evaluating
    different parameter combinations through simulation runs. It supports multiple
    task types with different optimization objectives and constraints.
    
    Attributes:
        task_type (str): Type of task ('hover', 'tracking', 'aggressive')
        drone_model (DroneModel): Drone model for simulation
        max_tilt_angle_deg (float): Maximum allowed tilt angle in degrees
        max_motor_output_pct (float): Maximum motor output percentage
        evaluation_count (int): Number of evaluations performed
        best_score (float): Best score achieved so far
        best_params (dict): Best parameters found
        evaluation_history (list): History of all evaluations
    """
    
    def __init__(self, task_type="hover", drone_model=DroneModel.CF2P, max_tilt_angle_deg=25, max_motor_output_pct=80):
        """
        Initialize PID tuner with task-specific parameters.
        
        Args:
            task_type (str): Task type ('hover', 'tracking', 'aggressive')
            drone_model (DroneModel): Drone model type
            max_tilt_angle_deg (float): Maximum tilt angle limit in degrees
            max_motor_output_pct (float): Maximum motor output percentage
        """
        self.task_type = task_type
        self.drone_model = drone_model
        self.max_tilt_angle_deg = max_tilt_angle_deg
        self.max_motor_output_pct = max_motor_output_pct
        self.evaluation_count = 0
        self.best_score = float('inf')
        self.best_params = None
        self.evaluation_history = []
        
        self._setup_search_bounds()
        self._setup_optimization_weights()
        
        print(f"PIDTuner initialized for {task_type} task")
        print(f"Safety limits: max_tilt={max_tilt_angle_deg}°, max_motor={max_motor_output_pct}%")
        
    def _setup_search_bounds(self):
        """Set up parameter search bounds based on task type."""
        
        if self.task_type == "hover":
            # Hover task: stability-focused, conservative parameter ranges
            self.param_bounds = {
                'pos_p': [(0.2, 0.8), (0.2, 0.8), (0.4, 1.0)],
                'pos_i': [(0.01, 0.15), (0.01, 0.15), (0.05, 0.2)],
                'pos_d': [(0.1, 0.5), (0.1, 0.5), (0.15, 0.6)],
                'att_p': [(5000, 15000), (5000, 15000), (4000, 12000)],
                'att_i': [(0.5, 8.0), (0.5, 8.0), (2.0, 15.0)],
                'att_d': [(400, 1500), (400, 1500), (300, 1200)]
            }
        elif self.task_type == "tracking":
            # Trajectory tracking: balance between responsiveness and precision
            self.param_bounds = {
                'pos_p': [(0.4, 1.2), (0.4, 1.2), (0.6, 1.4)],
                'pos_i': [(0.02, 0.2), (0.02, 0.2), (0.08, 0.25)],
                'pos_d': [(0.2, 0.8), (0.2, 0.8), (0.25, 0.9)],
                'att_p': [(7000, 20000), (7000, 20000), (6000, 18000)],
                'att_i': [(1.0, 12.0), (1.0, 12.0), (3.0, 20.0)],
                'att_d': [(600, 2000), (600, 2000), (500, 1800)]
            }
        elif self.task_type == "aggressive":
            # Aggressive task: responsiveness-focused
            self.param_bounds = {
                'pos_p': [(0.6, 1.5), (0.6, 1.5), (0.8, 1.8)],
                'pos_i': [(0.05, 0.3), (0.05, 0.3), (0.1, 0.4)],
                'pos_d': [(0.3, 1.0), (0.3, 1.0), (0.4, 1.2)],
                'att_p': [(10000, 30000), (10000, 30000), (8000, 25000)],
                'att_i': [(2.0, 20.0), (2.0, 20.0), (5.0, 30.0)],
                'att_d': [(800, 2500), (800, 2500), (700, 2200)]
            }
        else:
            # Default to hover parameters
            self.task_type = "hover"
            self._setup_search_bounds()
    
    def _setup_optimization_weights(self):
        """Set up multi-objective optimization weights based on task type."""
        if self.task_type == "hover":
            self.weights = {
                'stability': 0.4,
                'tracking_error': 0.3, 
                'overshoot': 0.2,
                'energy_efficiency': 0.1
            }
        elif self.task_type == "tracking":
            self.weights = {
                'stability': 0.25,
                'tracking_error': 0.35,
                'overshoot': 0.2,
                'energy_efficiency': 0.2
            }
        elif self.task_type == "aggressive":
            self.weights = {
                'stability': 0.2,
                'tracking_error': 0.3,
                'overshoot': 0.15,
                'energy_efficiency': 0.35
            }
    
    def objective_function(self, x):
        """
        Objective function for PID parameter evaluation.
        
        Args:
            x (array): PID parameter vector [pos_p(3), pos_i(3), pos_d(3), att_p(3), att_i(3), att_d(3)]
        
        Returns:
            float: Performance score (lower is better)
        """
        try:
            params = self._vector_to_params(x)
            score, detailed_scores = self._run_simulation_test(params)
            
            self.evaluation_count += 1
            
            self.evaluation_history.append({
                'evaluation': self.evaluation_count,
                'params': params,
                'score': score,
                'detailed_scores': detailed_scores,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"Evaluation {self.evaluation_count}: Score = {score:.4f}")
            # print(f"  Details: {detailed_scores}")  # Debug info - can be uncommented if needed
            
            if score < self.best_score and np.isfinite(score):
                self.best_score = score
                self.best_params = params.copy()
                print(f"New best score: {score:.4f}")
                self._save_best_params_to_json()
            
            return score
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return float('inf')
    
    def _run_simulation_test(self, params):
        """
        Run simulation test and evaluate performance.
        
        Args:
            params (dict): PID parameters dictionary
        
        Returns:
            tuple: (total_score, detailed_scores_dict)
        """
        try:
            temp_folder = f"temp_tuning_{int(time.time() * 1000000)}"
            
            run(
                duration_sec=8,  # Short test duration for faster tuning
                output_folder=temp_folder,
                gui=False,
                plot=False,
                custom_pid_params=params,
                randomize_positions=True,
                drone=self.drone_model,
                max_tilt_angle_deg=self.max_tilt_angle_deg,
                max_motor_output_pct=self.max_motor_output_pct
            )
            
            total_score, detailed_scores = self._analyze_performance(temp_folder)
            self._cleanup_temp_folder(temp_folder)
            
            return total_score, detailed_scores
            
        except Exception as e:
            print(f"Simulation test failed: {e}")
            return float('inf'), {}
    
    def _analyze_performance(self, log_folder):
        """
        Analyze simulation performance metrics.
        
        Args:
            log_folder (str): Path to log folder containing simulation data
        
        Returns:
            tuple: (total_score, detailed_scores_dict)
        """
        try:
            csv_folders = [f for f in os.listdir(log_folder) if f.startswith('save-flight-pid-')]
            
            if not csv_folders:
                return float('inf'), {}
            
            csv_folder = os.path.join(log_folder, csv_folders[0])
            data = self._load_csv_data(csv_folder)
            
            if data is None or len(data) < 50:
                return float('inf'), {}
            
            scores = {}
            scores['stability'] = self._evaluate_stability(data)
            scores['tracking_error'] = self._evaluate_tracking_error(data)
            scores['overshoot'] = self._evaluate_overshoot(data)
            scores['energy_efficiency'] = self._evaluate_energy_efficiency(data)
            
            total_score = sum(self.weights[key] * scores[key] for key in scores.keys())
            
            return total_score, scores
            
        except Exception as e:
            print(f"Performance analysis failed: {e}")
            return float('inf'), {}
    
    def _load_csv_data(self, csv_folder):
        """Load CSV simulation data from folder."""
        try:
            data = {}
            required_files = ['x0.csv', 'y0.csv', 'z0.csv', 'vx0.csv', 'vy0.csv', 'vz0.csv']
            
            for file in required_files:
                file_path = os.path.join(csv_folder, file)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, header=None)
                    key = file.replace('0.csv', '')
                    data[key] = df.iloc[:, 1].values
            
            if len(data) < 6:
                return None
                
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Failed to load CSV data: {e}")
            return None
    
    def _evaluate_stability(self, data):
        """Evaluate system stability using position and velocity variance."""
        try:
            # Use last 30% of data to evaluate steady-state stability
            stable_region = data.iloc[-int(len(data)*0.3):]
            
            pos_variance = np.var(stable_region[['x', 'y', 'z']].values, axis=0).mean()
            vel_variance = np.var(stable_region[['vx', 'vy', 'vz']].values, axis=0).mean()
            
            stability_score = np.tanh(pos_variance * 50 + vel_variance * 5)
            return stability_score
            
        except:
            return 1.0
    
    def _evaluate_tracking_error(self, data):
        """Evaluate tracking error relative to target position."""
        try:
            # Assume target position is final converged position
            target_pos = data[['x', 'y', 'z']].iloc[-10:].mean().values
            positions = data[['x', 'y', 'z']].values
            errors = np.linalg.norm(positions - target_pos, axis=1)
            mean_error = np.mean(errors)
            
            error_score = np.tanh(mean_error * 5)
            return error_score
            
        except:
            return 1.0
    
    def _evaluate_overshoot(self, data):
        """Evaluate overshoot amount."""
        try:
            z_positions = data['z'].values
            target_z = z_positions[-10:].mean()
            max_overshoot = np.max(np.abs(z_positions - target_z))
            
            overshoot_score = np.tanh(max_overshoot * 2)
            return overshoot_score
            
        except:
            return 1.0
    
    def _evaluate_energy_efficiency(self, data):
        """Evaluate energy efficiency based on velocity changes."""
        try:
            velocities = data[['vx', 'vy', 'vz']].values
            vel_changes = np.diff(velocities, axis=0)
            energy_metric = np.mean(np.sum(np.abs(vel_changes), axis=1))
            
            energy_score = np.tanh(energy_metric * 0.5)
            return energy_score
            
        except:
            return 0.5
    
    def _cleanup_temp_folder(self, folder_path):
        """Clean up temporary folder."""
        try:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
        except:
            pass
    
    def _vector_to_params(self, x):
        """Convert optimization vector to parameter dictionary."""
        return {
            'name': f'Tuned_{self.task_type}_{self.evaluation_count}',
            'pos_p': [x[0], x[1], x[2]],
            'pos_i': [x[3], x[4], x[5]], 
            'pos_d': [x[6], x[7], x[8]],
            'att_p': [x[9], x[10], x[11]],
            'att_i': [x[12], x[13], x[14]],
            'att_d': [x[15], x[16], x[17]]
        }
    
    def _params_to_vector(self, params):
        """Convert parameter dictionary to optimization vector."""
        return np.array([
            *params['pos_p'], *params['pos_i'], *params['pos_d'],
            *params['att_p'], *params['att_i'], *params['att_d']
        ])
    
    def _get_bounds_list(self):
        """Get search bounds as list for optimization."""
        bounds = []
        for key in ['pos_p', 'pos_i', 'pos_d', 'att_p', 'att_i', 'att_d']:
            bounds.extend(self.param_bounds[key])
        return bounds
    
    def optimize_differential_evolution(self, max_evaluations=1, population_size=1):
        """
        Optimize PID parameters using differential evolution algorithm.
        
        Args:
            max_evaluations (int): Maximum number of evaluations
            population_size (int): Population size for the algorithm
            
        Returns:
            OptimizeResult: Scipy optimization result object
        """
        print(f"Starting Differential Evolution optimization")
        print(f"Max evaluations: {max_evaluations}")
        print(f"Population size: {population_size}")
        
        bounds = self._get_bounds_list()
        
        self.evaluation_count = 0
        self.best_score = float('inf')
        self.best_params = None
        self.evaluation_history = []
        
        # Calculate maxiter safely to avoid infinite loops
        maxiter = max(1, max_evaluations // population_size)
        print(f"Calculated maxiter: {maxiter}")
        
        try:
            result = differential_evolution(
                func=self.objective_function,
                bounds=bounds,
                maxiter=maxiter,
                popsize=population_size,
                seed=42,
                disp=True,
                polish=False,
                atol=1e-3,
                tol=1e-3,
                workers=1,  # Use single worker to avoid potential issues
                updating='immediate'  # Use immediate updating strategy
            )
            
            print(f"\nOptimization completed!")
            print(f"Best score: {result.fun:.4f}")
            print(f"Total evaluations: {self.evaluation_count}")
            print(f"Optimization success: {result.success}")
            print(f"Optimization message: {result.message}")
            
            self._save_optimization_results(result)
            
            return result
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Optimization interrupted by user!")
            print(f"Evaluations completed: {self.evaluation_count}")
            if self.best_params:
                print(f"Best score so far: {self.best_score:.4f}")
                self._save_best_params_to_json()
            return None
        except Exception as e:
            print(f"Optimization failed: {e}")
            return None
    
    def _save_best_params_to_json(self):
        """Save best parameters to JSON file."""
        if self.best_params is None:
            return
            
        try:
            best_params_dir = os.path.join(
                os.path.dirname(__file__), 
                'Best_PID_Params'
            )
            os.makedirs(best_params_dir, exist_ok=True)
            
            best_params_file = os.path.join(best_params_dir, 'best_pid_params.json')
            
            save_data = {
                'name': f'Optimized_{self.task_type}_PID',
                'task_type': self.task_type,
                'drone_model': self.drone_model.name,
                'optimization_score': self.best_score,
                'evaluation_count': self.evaluation_count,
                'pos_p': self.best_params['pos_p'],
                'pos_i': self.best_params['pos_i'],
                'pos_d': self.best_params['pos_d'],
                'att_p': self.best_params['att_p'],
                'att_i': self.best_params['att_i'],
                'att_d': self.best_params['att_d']
            }
            
            with open(best_params_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            print(f"Best parameters saved to: {best_params_file}")
            
        except Exception as e:
            print(f"Failed to save best parameters: {e}")
    
    def _save_optimization_results(self, result):
        """Save complete optimization results to file."""
        try:
            results_dir = os.path.join(os.path.dirname(__file__), 'tuning_results')
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = os.path.join(results_dir, f'pid_tuning_results_{timestamp}.json')
            
            results_data = {
                'task_type': self.task_type,
                'drone_model': self.drone_model.name,
                'optimization_method': 'differential_evolution',
                'best_score': self.best_score,
                'best_params': self.best_params,
                'total_evaluations': self.evaluation_count,
                'optimization_weights': self.weights,
                'search_bounds': self.param_bounds,
                'evaluation_history': self.evaluation_history[-10:],  # Keep last 10 evaluations
                'scipy_result': {
                    'success': result.success,
                    'message': result.message,
                    'fun': result.fun,
                    'nfev': result.nfev
                } if result else None,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            print(f"Full results saved to: {results_file}")
            
        except Exception as e:
            print(f"Failed to save optimization results: {e}")
    
    def plot_optimization_history(self):
        """Plot optimization history and performance metrics."""
        if not self.evaluation_history:
            print("No optimization history to plot")
            return
            
        try:
            evaluations = [h['evaluation'] for h in self.evaluation_history]
            scores = [h['score'] for h in self.evaluation_history]
            
            plt.figure(figsize=(12, 8))
            
            # Main plot: score progression
            plt.subplot(2, 2, 1)
            plt.plot(evaluations, scores, 'b-', alpha=0.7, label='Scores')
            plt.axhline(y=self.best_score, color='r', linestyle='--', label=f'Best: {self.best_score:.4f}')
            plt.xlabel('Evaluation')
            plt.ylabel('Score')
            plt.title('Optimization Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplots: detailed metric progression
            detailed_metrics = ['stability', 'tracking_error', 'overshoot', 'energy_efficiency']
            
            for i, metric in enumerate(detailed_metrics, 2):
                plt.subplot(2, 2, i)
                metric_values = []
                for h in self.evaluation_history:
                    if 'detailed_scores' in h and metric in h['detailed_scores']:
                        metric_values.append(h['detailed_scores'][metric])
                    else:
                        metric_values.append(np.nan)
                
                plt.plot(evaluations[:len(metric_values)], metric_values, 'g-', alpha=0.7)
                plt.xlabel('Evaluation')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.title(f'{metric.replace("_", " ").title()} Progress')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plots_dir = os.path.join(os.path.dirname(__file__), 'tuning_plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = os.path.join(plots_dir, f'pid_tuning_history_{timestamp}.png')
            
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Optimization history plot saved to: {plot_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"Failed to plot optimization history: {e}")

def main():
    """
    Main function for command-line interface.
    
    Provides command-line argument parsing and runs the PID tuning process
    with specified parameters and options.
    """
    parser = argparse.ArgumentParser(description='PID Parameter Tuner')
    parser.add_argument('--task', default='hover', choices=['hover', 'tracking', 'aggressive'], 
                       help='Task type for optimization')
    parser.add_argument('--drone', default='cf2p', choices=['cf2x', 'cf2p'], 
                       help='Drone model')
    parser.add_argument('--max_eval', default=60, type=int, 
                       help='Maximum number of evaluations')
    parser.add_argument('--population', default=10, type=int, 
                       help='Population size for differential evolution')
    parser.add_argument('--plot', action='store_true', 
                       help='Plot optimization history')
    parser.add_argument('--max_tilt_angle_deg', default=25, type=float,
                       help='Maximum tilt angle in degrees (default: 25)')
    parser.add_argument('--max_motor_output_pct', default=80, type=float,
                       help='Maximum motor output percentage (default: 80)')
    
    args = parser.parse_args()
    
    drone_model = DroneModel.CF2P if args.drone == 'cf2p' else DroneModel.CF2X
    
    print("PID Parameter Tuner")
    print("=" * 50)
    print(f"Task: {args.task}")
    print(f"Drone: {drone_model.name}")
    print(f"Max evaluations: {args.max_eval}")
    print(f"Population size: {args.population}")
    print(f"Max tilt angle: {args.max_tilt_angle_deg}°")
    print(f"Max motor output: {args.max_motor_output_pct}%")
    print("=" * 50)
    
    tuner = PIDTuner(
        task_type=args.task, 
        drone_model=drone_model,
        max_tilt_angle_deg=args.max_tilt_angle_deg,
        max_motor_output_pct=args.max_motor_output_pct
    )
    
    result = tuner.optimize_differential_evolution(
        max_evaluations=args.max_eval,
        population_size=args.population
    )
    
    if result and result.success:
        print("\nOptimization successful!")
        print(f"Best parameters saved as best_pid_params.json")
        print(f"Best score: {tuner.best_score:.4f}")
        
        if tuner.best_params:
            print("\nBest Parameters:")
            for key, value in tuner.best_params.items():
                if key != 'name':
                    print(f"  {key}: {value}")
    else:
        print("\nOptimization failed or interrupted")
    
    if args.plot:
        tuner.plot_optimization_history()

if __name__ == "__main__":
    main()
