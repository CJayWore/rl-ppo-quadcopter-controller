#!/usr/bin/env python3
"""
PID参数自动调优器
自动搜索最优PID参数，并将结果保存为best_pid_params.json
"""

import numpy as np
import json
import os
import sys
import time
import argparse
import shutil
from datetime import datetime
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import pandas as pd

# 添加当前目录到路径以便导入pid模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入无人机仿真模块
from pid import run
from gym_pybullet_drones.utils.enums import DroneModel

class PIDTuner:
    """PID参数调优器"""
    
    def __init__(self, task_type="hover", drone_model=DroneModel.CF2P, max_tilt_angle_deg=25, max_motor_output_pct=80):
        """
        初始化PID调优器
        
        Parameters:
        task_type: str - 任务类型 ('hover', 'tracking', 'aggressive')
        drone_model: DroneModel - 无人机类型
        max_tilt_angle_deg: float - 最大倾角限制（度）
        max_motor_output_pct: float - 最大电机输出百分比
        """
        self.task_type = task_type
        self.drone_model = drone_model
        self.max_tilt_angle_deg = max_tilt_angle_deg
        self.max_motor_output_pct = max_motor_output_pct
        self.evaluation_count = 0
        self.best_score = float('inf')
        self.best_params = None
        self.evaluation_history = []
        
        # 设置搜索空间
        self._setup_search_bounds()
        
        # 设置优化权重
        self._setup_optimization_weights()
        
        print(f"🎯 PIDTuner initialized for {task_type} task")
        print(f"🛡️ Safety limits: max_tilt={max_tilt_angle_deg}°, max_motor={max_motor_output_pct}%")
        
    def _setup_search_bounds(self):
        """根据任务类型设置合理的搜索边界"""
        
        if self.task_type == "hover":
            # 悬停任务：稳定性优先，保守参数范围
            self.param_bounds = {
                'pos_p': [(0.2, 0.8), (0.2, 0.8), (0.4, 1.0)],
                'pos_i': [(0.01, 0.15), (0.01, 0.15), (0.05, 0.2)],
                'pos_d': [(0.1, 0.5), (0.1, 0.5), (0.15, 0.6)],
                'att_p': [(5000, 15000), (5000, 15000), (4000, 12000)],
                'att_i': [(0.5, 8.0), (0.5, 8.0), (2.0, 15.0)],
                'att_d': [(400, 1500), (400, 1500), (300, 1200)]
            }
        elif self.task_type == "tracking":
            # 轨迹跟踪：响应性和精度并重
            self.param_bounds = {
                'pos_p': [(0.4, 1.2), (0.4, 1.2), (0.6, 1.4)],
                'pos_i': [(0.02, 0.2), (0.02, 0.2), (0.08, 0.25)],
                'pos_d': [(0.2, 0.8), (0.2, 0.8), (0.25, 0.9)],
                'att_p': [(7000, 20000), (7000, 20000), (6000, 18000)],
                'att_i': [(1.0, 12.0), (1.0, 12.0), (3.0, 20.0)],
                'att_d': [(600, 2000), (600, 2000), (500, 1800)]
            }
        elif self.task_type == "aggressive":
            # 激进任务：响应速度优先
            self.param_bounds = {
                'pos_p': [(0.6, 1.5), (0.6, 1.5), (0.8, 1.8)],
                'pos_i': [(0.05, 0.3), (0.05, 0.3), (0.1, 0.4)],
                'pos_d': [(0.3, 1.0), (0.3, 1.0), (0.4, 1.2)],
                'att_p': [(10000, 30000), (10000, 30000), (8000, 25000)],
                'att_i': [(2.0, 20.0), (2.0, 20.0), (5.0, 30.0)],
                'att_d': [(800, 2500), (800, 2500), (700, 2200)]
            }
        else:
            # 默认使用悬停参数
            self.task_type = "hover"
            self._setup_search_bounds()
    
    def _setup_optimization_weights(self):
        """设置多目标优化权重"""
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
        目标函数：评估PID参数的性能
        
        Parameters:
        x: array - PID参数向量 [pos_p(3), pos_i(3), pos_d(3), att_p(3), att_i(3), att_d(3)]
        
        Returns:
        float - 性能评分（越小越好）
        """
        try:
            # 将向量转换为参数字典
            params = self._vector_to_params(x)
            
            # 运行仿真测试
            score, detailed_scores = self._run_simulation_test(params)
            
            self.evaluation_count += 1
            
            # 记录评估历史
            self.evaluation_history.append({
                'evaluation': self.evaluation_count,
                'params': params,
                'score': score,
                'detailed_scores': detailed_scores,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"Evaluation {self.evaluation_count}: Score = {score:.4f}")
            print(f"  Details: {detailed_scores}")
            
            # 更新最佳结果
            if score < self.best_score and np.isfinite(score):
                self.best_score = score
                self.best_params = params.copy()
                print(f"🏆 New best score: {score:.4f}")
                
                # 实时保存最佳参数
                self._save_best_params_to_json()
            
            return score
            
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return float('inf')
    
    def _run_simulation_test(self, params):
        """
        运行仿真测试并评估性能
        
        Parameters:
        params: dict - PID参数字典
        
        Returns:
        tuple: (总分, 详细分数字典)
        """
        try:
            # 创建临时输出文件夹
            temp_folder = f"temp_tuning_{int(time.time() * 1000000)}"
            
            # 运行仿真
            run(
                duration_sec=8,  # 短时间测试以加快调优速度
                output_folder=temp_folder,
                gui=False,
                plot=False,
                custom_pid_params=params,
                randomize_positions=True,
                drone=self.drone_model,
                max_tilt_angle_deg=self.max_tilt_angle_deg,
                max_motor_output_pct=self.max_motor_output_pct
            )
            
            # 分析结果
            total_score, detailed_scores = self._analyze_performance(temp_folder)
            
            # 清理临时文件
            self._cleanup_temp_folder(temp_folder)
            
            return total_score, detailed_scores
            
        except Exception as e:
            print(f"❌ Simulation test failed: {e}")
            return float('inf'), {}
    
    def _analyze_performance(self, log_folder):
        """
        分析仿真结果性能
        
        Parameters:
        log_folder: str - 日志文件夹路径
        
        Returns:
        tuple: (总分, 详细分数字典)
        """
        try:
            # 查找CSV数据文件夹
            csv_folders = [f for f in os.listdir(log_folder) if f.startswith('save-flight-pid-')]
            
            if not csv_folders:
                return float('inf'), {}
            
            csv_folder = os.path.join(log_folder, csv_folders[0])
            
            # 加载仿真数据
            data = self._load_csv_data(csv_folder)
            
            if data is None or len(data) < 50:
                return float('inf'), {}
            
            # 计算各项性能指标
            scores = {}
            
            # 1. 稳定性评估（位置方差）
            scores['stability'] = self._evaluate_stability(data)
            
            # 2. 跟踪误差评估
            scores['tracking_error'] = self._evaluate_tracking_error(data)
            
            # 3. 超调量评估
            scores['overshoot'] = self._evaluate_overshoot(data)
            
            # 4. 能效评估
            scores['energy_efficiency'] = self._evaluate_energy_efficiency(data)
            
            # 计算加权总分
            total_score = sum(self.weights[key] * scores[key] for key in scores.keys())
            
            return total_score, scores
            
        except Exception as e:
            print(f"❌ Performance analysis failed: {e}")
            return float('inf'), {}
    
    def _load_csv_data(self, csv_folder):
        """加载CSV仿真数据"""
        try:
            data = {}
            required_files = ['x0.csv', 'y0.csv', 'z0.csv', 'vx0.csv', 'vy0.csv', 'vz0.csv']
            
            for file in required_files:
                file_path = os.path.join(csv_folder, file)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, header=None)
                    key = file.replace('0.csv', '')
                    data[key] = df.iloc[:, 1].values  # 第二列是数据值
            
            if len(data) < 6:  # 确保有基本的位置和速度数据
                return None
                
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"❌ Failed to load CSV data: {e}")
            return None
    
    def _evaluate_stability(self, data):
        """评估系统稳定性"""
        try:
            # 使用最后30%的数据评估稳定性
            stable_region = data.iloc[-int(len(data)*0.3):]
            
            # 计算位置方差（越小越稳定）
            pos_variance = np.var(stable_region[['x', 'y', 'z']].values, axis=0).mean()
            
            # 计算速度方差（越小越稳定）
            vel_variance = np.var(stable_region[['vx', 'vy', 'vz']].values, axis=0).mean()
            
            # 归一化评分
            stability_score = np.tanh(pos_variance * 50 + vel_variance * 5)
            
            return stability_score
            
        except:
            return 1.0  # 最差评分
    
    def _evaluate_tracking_error(self, data):
        """评估跟踪误差"""
        try:
            # 假设目标位置是最终收敛位置
            target_pos = data[['x', 'y', 'z']].iloc[-10:].mean().values
            
            # 计算整体跟踪误差
            positions = data[['x', 'y', 'z']].values
            errors = np.linalg.norm(positions - target_pos, axis=1)
            mean_error = np.mean(errors)
            
            # 归一化评分
            error_score = np.tanh(mean_error * 5)
            
            return error_score
            
        except:
            return 1.0
    
    def _evaluate_overshoot(self, data):
        """评估超调量"""
        try:
            # 简化评估：使用Z轴位置变化
            z_positions = data['z'].values
            target_z = z_positions[-10:].mean()
            
            # 计算最大超调
            max_overshoot = np.max(np.abs(z_positions - target_z))
            
            # 归一化评分
            overshoot_score = np.tanh(max_overshoot * 2)
            
            return overshoot_score
            
        except:
            return 1.0
    
    def _evaluate_energy_efficiency(self, data):
        """评估能源效率"""
        try:
            # 使用速度变化率作为能耗指标
            velocities = data[['vx', 'vy', 'vz']].values
            
            # 计算速度变化的总量
            vel_changes = np.diff(velocities, axis=0)
            energy_metric = np.mean(np.sum(np.abs(vel_changes), axis=1))
            
            # 归一化评分
            energy_score = np.tanh(energy_metric * 0.5)
            
            return energy_score
            
        except:
            return 0.5  # 中等评分
    
    def _cleanup_temp_folder(self, folder_path):
        """清理临时文件夹"""
        try:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
        except:
            pass  # 忽略清理错误
    
    def _vector_to_params(self, x):
        """将优化向量转换为参数字典"""
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
        """将参数字典转换为优化向量"""
        return np.array([
            *params['pos_p'], *params['pos_i'], *params['pos_d'],
            *params['att_p'], *params['att_i'], *params['att_d']
        ])
    
    def _get_bounds_list(self):
        """获取搜索边界列表"""
        bounds = []
        for key in ['pos_p', 'pos_i', 'pos_d', 'att_p', 'att_i', 'att_d']:
            bounds.extend(self.param_bounds[key])
        return bounds
    
    def optimize_differential_evolution(self, max_evaluations=1000, population_size=10):
        """使用差分进化算法优化"""
        print(f"🚀 Starting Differential Evolution optimization")
        print(f"📊 Max evaluations: {max_evaluations}")
        print(f"👥 Population size: {population_size}")
        
        bounds = self._get_bounds_list()
        
        # 重置计数器
        self.evaluation_count = 0
        self.best_score = float('inf')
        self.best_params = None
        self.evaluation_history = []
        
        try:
            result = differential_evolution(
                func=self.objective_function,
                bounds=bounds,
                maxiter=max_evaluations // population_size,
                popsize=population_size,
                seed=42,
                disp=True,
                polish=False,
                atol=1e-3,
                tol=1e-3
            )
            
            print(f"\n🎉 Optimization completed!")
            print(f"🏆 Best score: {result.fun:.4f}")
            print(f"📈 Total evaluations: {self.evaluation_count}")
            
            # 保存最终结果
            self._save_optimization_results(result)
            
            return result
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return None
    
    def _save_best_params_to_json(self):
        """保存最佳参数到JSON文件"""
        if self.best_params is None:
            return
            
        try:
            # 创建Best_PID_Params文件夹
            best_params_dir = os.path.join(
                os.path.dirname(__file__), 
                'Best_PID_Params'
            )
            os.makedirs(best_params_dir, exist_ok=True)
            
            # 保存到best_pid_params.json
            best_params_file = os.path.join(best_params_dir, 'best_pid_params.json')
            
            # 准备保存的数据 - 只包含必要的PID参数，不包含时间戳
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
            
            print(f"💾 Best parameters saved to: {best_params_file}")
            
        except Exception as e:
            print(f"❌ Failed to save best parameters: {e}")
    
    def _save_optimization_results(self, result):
        """保存完整的优化结果"""
        try:
            # 创建结果文件夹
            results_dir = os.path.join(os.path.dirname(__file__), 'tuning_results')
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = os.path.join(results_dir, f'pid_tuning_results_{timestamp}.json')
            
            # 准备完整结果数据
            results_data = {
                'task_type': self.task_type,
                'drone_model': self.drone_model.name,
                'optimization_method': 'differential_evolution',
                'best_score': self.best_score,
                'best_params': self.best_params,
                'total_evaluations': self.evaluation_count,
                'optimization_weights': self.weights,
                'search_bounds': self.param_bounds,
                'evaluation_history': self.evaluation_history[-10:],  # 只保存最后10次评估
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
            
            print(f"📁 Full results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Failed to save optimization results: {e}")
    
    def plot_optimization_history(self):
        """绘制优化历史"""
        if not self.evaluation_history:
            print("❌ No optimization history to plot")
            return
            
        try:
            # 提取数据
            evaluations = [h['evaluation'] for h in self.evaluation_history]
            scores = [h['score'] for h in self.evaluation_history]
            
            # 创建图表
            plt.figure(figsize=(12, 8))
            
            # 主图：总分变化
            plt.subplot(2, 2, 1)
            plt.plot(evaluations, scores, 'b-', alpha=0.7, label='Scores')
            plt.axhline(y=self.best_score, color='r', linestyle='--', label=f'Best: {self.best_score:.4f}')
            plt.xlabel('Evaluation')
            plt.ylabel('Score')
            plt.title('Optimization Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图：详细指标变化
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
            
            # 保存图表
            plots_dir = os.path.join(os.path.dirname(__file__), 'tuning_plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = os.path.join(plots_dir, f'pid_tuning_history_{timestamp}.png')
            
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Optimization history plot saved to: {plot_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Failed to plot optimization history: {e}")

def main():
    """主函数"""
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
    
    # 设置无人机模型
    drone_model = DroneModel.CF2P if args.drone == 'cf2p' else DroneModel.CF2X
    
    print("🎯 PID Parameter Tuner")
    print("=" * 50)
    print(f"Task: {args.task}")
    print(f"Drone: {drone_model.name}")
    print(f"Max evaluations: {args.max_eval}")
    print(f"Population size: {args.population}")
    print(f"🛡️ Max tilt angle: {args.max_tilt_angle_deg}°")
    print(f"🛡️ Max motor output: {args.max_motor_output_pct}%")
    print("=" * 50)
    
    # 创建调优器
    tuner = PIDTuner(
        task_type=args.task, 
        drone_model=drone_model,
        max_tilt_angle_deg=args.max_tilt_angle_deg,
        max_motor_output_pct=args.max_motor_output_pct
    )
    
    # 运行优化
    result = tuner.optimize_differential_evolution(
        max_evaluations=args.max_eval,
        population_size=args.population
    )
    
    if result and result.success:
        print("\n🎉 Optimization successful!")
        print(f"🏆 Best parameters saved as best_pid_params.json")
        print(f"📊 Best score: {tuner.best_score:.4f}")
        
        if tuner.best_params:
            print("\n📋 Best Parameters:")
            for key, value in tuner.best_params.items():
                if key != 'name':
                    print(f"  {key}: {value}")
    else:
        print("\n❌ Optimization failed or interrupted")
    
    # 绘制优化历史
    if args.plot:
        tuner.plot_optimization_history()

if __name__ == "__main__":
    main()
