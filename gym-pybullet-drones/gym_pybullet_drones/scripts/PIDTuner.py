import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from pid import run
import os
import json
from datetime import datetime
import time
import shutil
import glob

class PIDTuner:
    def __init__(self):
        self.test_results = []
        self.best_params = None
        self.best_score = float('inf')
        self.evaluation_count = 0
        
        # PID参数的搜索范围
        self.param_bounds = {
            'pos_p': [(0.1, 1.0), (0.1, 1.0), (0.5, 2.5)],  # [x, y, z]
            'pos_i': [(0.001, 0.2), (0.001, 0.2), (0.001, 0.2)],
            'pos_d': [(0.05, 0.8), (0.05, 0.8), (0.1, 1.5)],
            'att_p': [(10000, 50000), (10000, 50000), (8000, 40000)],
            'att_i': [(0, 20), (0, 20), (50, 500)], 
            'att_d': [(2000, 15000), (2000, 15000), (1500, 10000)]
        }
    
    def objective_function(self, x):
        """优化目标函数"""
        try:
            # 将参数向量转换为参数字典
            params = self._vector_to_params(x)
            
            # 评估参数性能
            score = self._run_and_evaluate(params, trajectory="circle", duration=6)
            
            self.evaluation_count += 1
            print(f"Evaluation {self.evaluation_count}: Score = {score:.4f}")
            
            # 记录结果
            if score < self.best_score and np.isfinite(score):
                self.best_score = score
                self.best_params = params
                print(f"🎯 New best score: {score:.4f}")
            
            return score
            
        except Exception as e:
            print(f"❌ Objective function error: {e}")
            return float('inf')
    
    def _vector_to_params(self, x):
        """将优化向量转换为PID参数字典"""
        return {
            'name': f'Optimized_{self.evaluation_count}',
            'pos_p': [x[0], x[1], x[2]],
            'pos_i': [x[3], x[4], x[5]],
            'pos_d': [x[6], x[7], x[8]],
            'att_p': [x[9], x[10], x[11]],
            'att_i': [x[12], x[13], x[14]],
            'att_d': [x[15], x[16], x[17]]
        }
    
    def _params_to_vector(self, params):
        """将PID参数字典转换为优化向量"""
        return np.array([
            *params['pos_p'], *params['pos_i'], *params['pos_d'],
            *params['att_p'], *params['att_i'], *params['att_d']
        ])
    
    def _get_bounds_vector(self):
        """获取所有参数的边界"""
        bounds = []
        for key in ['pos_p', 'pos_i', 'pos_d', 'att_p', 'att_i', 'att_d']:
            bounds.extend(self.param_bounds[key])
        return bounds
    
    def _run_and_evaluate(self, params, trajectory, duration):
        """运行仿真并评估性能"""
        try:
            # 创建唯一的临时文件夹
            temp_folder = f"temp_results_{int(time.time() * 1000000)}"
            
            # 确保文件夹不存在
            while os.path.exists(temp_folder):
                temp_folder = f"temp_results_{int(time.time() * 1000000)}"
                time.sleep(0.001)
            
            print(f"📁 Running simulation in: {temp_folder}")
            
            # 运行仿真
            run(
                trajectory=trajectory,
                duration_sec=duration,
                output_folder=temp_folder,
                gui=False,
                plot=False,
                custom_pid_params=params
            )
            
            # 计算性能分数
            score = self._calculate_performance_score(temp_folder)
            self._cleanup_temp_files(temp_folder)
            
            return score
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')
    
    def _calculate_performance_score(self, log_folder):
        """计算性能分数 - 根据Logger的实际保存格式"""
        try:
            print(f"📁 Analyzing log folder: {log_folder}")
            
            # 检查文件夹是否存在
            if not os.path.exists(log_folder):
                print(f"❌ Log folder does not exist: {log_folder}")
                return float('inf')
            
            # 列出文件夹中的所有文件
            files = os.listdir(log_folder)
            print(f"📁 Files in {log_folder}: {files}")
            
            # 查找CSV文件夹（由save_as_csv创建）
            csv_folders = [f for f in files if f.startswith('save-flight-pid-')]
            if csv_folders:
                # 使用最新的CSV文件夹
                csv_folder = os.path.join(log_folder, csv_folders[-1])
                print(f"📊 Found CSV folder: {csv_folder}")
                return self._calculate_score_from_csv_folder(csv_folder)
            
            # 查找NPY文件（由save创建）
            npy_files = [f for f in files if f.endswith('.npz')]
            if npy_files:
                npy_file = os.path.join(log_folder, npy_files[-1])
                print(f"📊 Found NPY file: {npy_file}")
                return self._calculate_score_from_npy(npy_file)
            
            print(f"❌ No valid log files found in {log_folder}")
            return float('inf')
            
        except Exception as e:
            print(f"❌ Score calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')
    
    def _calculate_score_from_csv_folder(self, csv_folder):
        """从CSV文件夹计算性能分数"""
        try:
            # 读取位置数据 (x0.csv, y0.csv, z0.csv)
            x_file = os.path.join(csv_folder, "x0.csv")
            y_file = os.path.join(csv_folder, "y0.csv")
            z_file = os.path.join(csv_folder, "z0.csv")
            
            if not all(os.path.exists(f) for f in [x_file, y_file, z_file]):
                print(f"❌ Missing position CSV files in {csv_folder}")
                return float('inf')
            
            # 读取位置数据
            x_data = pd.read_csv(x_file, header=None).values
            y_data = pd.read_csv(y_file, header=None).values
            z_data = pd.read_csv(z_file, header=None).values
            
            # 提取时间和位置
            t = x_data[:, 0]  # 时间
            pos_x = x_data[:, 1]  # x位置
            pos_y = y_data[:, 1]  # y位置
            pos_z = z_data[:, 1]  # z位置
            
            # 计算圆形轨迹的目标位置
            R = 0.3  # 半径
            PERIOD = 10  # 周期
            target_x = R * np.cos(2 * np.pi * t / PERIOD)
            target_y = R * np.sin(2 * np.pi * t / PERIOD) - R
            target_z = np.ones_like(t) * 0.1  # 目标高度
            
            # 计算位置误差
            pos_error = np.sqrt(
                (pos_x - target_x)**2 +
                (pos_y - target_y)**2 +
                (pos_z - target_z)**2
            )
            
            # 过滤无效值
            pos_error = pos_error[np.isfinite(pos_error)]
            
            if len(pos_error) == 0:
                print("❌ No valid position error data")
                return float('inf')
            
            # 计算性能指标
            avg_error = pos_error.mean()
            max_error = pos_error.max()
            std_error = pos_error.std()
            
            # 稳态误差（最后20%的数据）
            steady_start = int(len(pos_error) * 0.8)
            steady_error = pos_error[steady_start:].mean() if steady_start < len(pos_error) else avg_error
            
            # 综合得分
            score = (0.4 * avg_error + 
                    0.3 * max_error + 
                    0.2 * std_error + 
                    0.1 * steady_error)
            
            print(f"Performance - Avg: {avg_error:.4f}, Max: {max_error:.4f}, Std: {std_error:.4f}, Steady: {steady_error:.4f}")
            print(f"Final Score: {score:.4f}")
            
            return score if np.isfinite(score) else float('inf')
            
        except Exception as e:
            print(f"❌ CSV folder score calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')
    
    def _calculate_score_from_npy(self, npy_file):
        """从NPY文件计算性能分数"""
        try:
            # 加载NPY数据
            data = np.load(npy_file)
            timestamps = data['timestamps'][0]  # 第一个无人机的时间戳
            states = data['states'][0]  # 第一个无人机的状态 [16, N]
            controls = data['controls'][0]  # 第一个无人机的控制目标 [12, N]
            
            # 获取实际数据长度
            valid_length = int(np.count_nonzero(timestamps))
            
            # 提取位置数据
            pos_x = states[0, :valid_length]  # x位置
            pos_y = states[1, :valid_length]  # y位置
            pos_z = states[2, :valid_length]  # z位置
            
            # 提取控制目标位置
            target_x = controls[0, :valid_length]
            target_y = controls[1, :valid_length]
            target_z = controls[2, :valid_length]
            
            # 计算位置误差
            pos_error = np.sqrt(
                (pos_x - target_x)**2 +
                (pos_y - target_y)**2 +
                (pos_z - target_z)**2
            )
            
            # 过滤无效值
            pos_error = pos_error[np.isfinite(pos_error)]
            
            if len(pos_error) == 0:
                print("❌ No valid position error data")
                return float('inf')
            
            # 计算性能指标
            avg_error = pos_error.mean()
            max_error = pos_error.max()
            std_error = pos_error.std()
            
            # 稳态误差（最后20%的数据）
            steady_start = int(len(pos_error) * 0.8)
            steady_error = pos_error[steady_start:].mean() if steady_start < len(pos_error) else avg_error
            
            # 综合得分
            score = (0.4 * avg_error + 
                    0.3 * max_error + 
                    0.2 * std_error + 
                    0.1 * steady_error)
            
            print(f"📊 Performance - Avg: {avg_error:.4f}, Max: {max_error:.4f}, Std: {std_error:.4f}, Steady: {steady_error:.4f}")
            print(f"📊 Final Score: {score:.4f}")
            
            return score if np.isfinite(score) else float('inf')
            
        except Exception as e:
            print(f"❌ NPY score calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')
    
    def _cleanup_temp_files(self, folder):
        """清理临时文件"""
        try:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"🧹 Cleaned up: {folder}")
        except Exception as e:
            print(f"⚠️ Cleanup failed for {folder}: {e}")
    
    def optimize_genetic_algorithm(self, max_evaluations=500, population_size=6):
        """遗传算法优化（减少评估次数以加快速度）"""
        print("Genetic Algorithm optimization")
        print(f"Max evaluations: {max_evaluations}")
        print(f"Population size: {population_size}")
        
        bounds = self._get_bounds_vector()
        
        # 重置计数器
        self.evaluation_count = 0
        self.best_score = float('inf')
        self.best_params = None
        
        result = differential_evolution(
            func=self.objective_function,
            bounds=bounds,
            maxiter=max_evaluations // population_size,
            popsize=population_size,
            seed=13,
            disp=True,
            polish=False,
            atol=1e-3,
            tol=1e-3
        )
        
        print(f"\n✅ Optimization completed!")
        print(f"Best score: {result.fun:.4f}")
        print(f"Total evaluations: {self.evaluation_count}")
        
        return result
    
    def optimize_random_search(self, max_evaluations=500):
        """随机搜索优化"""
        print("Random Search optimization")
        
        bounds = self._get_bounds_vector()
        
        # 重置计数器
        self.evaluation_count = 0
        self.best_score = float('inf')
        self.best_params = None
        
        for i in range(max_evaluations):
            # 生成随机参数
            x = np.array([
                np.random.uniform(bound[0], bound[1]) 
                for bound in bounds
            ])
            
            # 评估参数
            self.objective_function(x)
        
        print(f"\n✅Random search completed!")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Total evaluations: {self.evaluation_count}")

    def optimize_bayesian(self, max_evaluations=500, acquisition_function='LCB'):
        from skopt import gp_minimize, forest_minimize, gbrt_minimize
        from skopt.space import Real
        from skopt.utils import use_named_args
        from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
        
        print("Bayesian optimization")
        print(f"Max evaluations: {max_evaluations}")
        print(f"Acquisition function: {acquisition_function}")
                
            
        # 定义搜索空间
        dimensions = []
        param_names = []

        for param_group in ['pos_p', 'pos_i', 'pos_d', 'att_p', 'att_i', 'att_d']:
            for i, bound in enumerate(self.param_bounds[param_group]):
                dimensions.append(Real(bound[0], bound[1], name=f'{param_group}_{i}'))
                param_names.append(f'{param_group}_{i}')
        
        print(f"🔍 Search space: {len(dimensions)} dimensions")
        for i, dim in enumerate(dimensions):
            print(f"  {param_names[i]}: [{dim.low:.4f}, {dim.high:.4f}]")
        
        # 定义目标函数（使用装饰器自动处理参数）
        @use_named_args(dimensions)
        def bayesian_objective(**params):
            """贝叶斯优化的目标函数"""
            # 将命名参数转换为向量
            x = []
            for name in param_names:
                x.append(params[name])
            
            return self.objective_function(np.array(x))
        # 重置计数器
        self.evaluation_count = 0
        self.best_score = float('inf')
        self.best_params = None
        
        # 选择采集函数
        acq_func_map = {
            'EI': 'EI',      # Expected Improvement
            'PI': 'PI',      # Probability of Improvement  
            'LCB': 'LCB',    # Lower Confidence Bound
            'gp_hedge': 'gp_hedge'  # 自动选择最佳采集函数
        }
        
        acq_func_name = acq_func_map.get(acquisition_function, 'EI')
        
        try:
            # 运行贝叶斯优化
            result = gp_minimize(
                func=bayesian_objective,
                dimensions=dimensions,
                n_calls=max_evaluations,
                n_initial_points=max(5, max_evaluations // 5),  # 初始随机点数
                acq_func=acq_func_name,
                n_jobs=1,  # 串行执行以避免仿真冲突
                random_state=42,
                verbose=True,
                noise=1e-10  # 添加小量噪声处理数值稳定性
            )
            
            print(f"\n✅Bayesian optimization completed!")
            print(f"Best score: {result.fun:.4f}")
            print(f"Total evaluations: {self.evaluation_count}")
            print(f"Convergence: {len(result.func_vals)} function evaluations")
            
            # 显示收敛历史
            print(f"\nConvergence history (last 5):")
            for i, score in enumerate(result.func_vals[-5:]):
                print(f"  Eval {len(result.func_vals)-4+i}: {score:.4f}")
            
            return result
        except Exception as e:
            print(f"❌ Bayesian optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    tuner = PIDTuner()
    
    # 选择优化方法
    print("Select an optimizer:")
    print("1. Genetic Algorithm")
    print("2. Random Search")
    print("3. Bayesian Optimization")
    
    choice = input("Select(1-3): ").strip()
    
    try:
        if choice == "1":
            result = tuner.optimize_genetic_algorithm(max_evaluations=12, population_size=4)
        elif choice == "2":
            tuner.optimize_random_search(max_evaluations=20)
        elif choice == "3":
            result = tuner.optimize_bayesian(max_evaluations=20, acquisition_function='LCB')

        # 保存结果
        if tuner.best_params:
            print(f"\nBest Parameters:")
            print(f"Position P={tuner.best_params['pos_p']}")
            print(f"Position I={tuner.best_params['pos_i']}")
            print(f"Position D={tuner.best_params['pos_d']}")
            print(f"Attitude P={tuner.best_params['att_p']}")
            print(f"Attitude I={tuner.best_params['att_i']}")
            print(f"Attitude D={tuner.best_params['att_d']}")
            
            # 保存到文件
            with open("best_pid_params.json", "w") as f:
                json.dump({
                    'best_score': tuner.best_score,
                    'best_params': tuner.best_params,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            print("Result is saved as best_pid_params.json")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()