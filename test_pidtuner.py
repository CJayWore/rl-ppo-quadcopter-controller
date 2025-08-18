#!/usr/bin/env python3
"""
PIDTuner演示脚本
展示如何使用PIDTuner进行参数优化
"""

import sys
import os

# 添加gym-pybullet-drones路径
gym_path = '/Users/cjaywore/Github_Repositories/24-25_CE901-SU_CE902-SP_chen_jiwei/gym-pybullet-drones'
if gym_path not in sys.path:
    sys.path.insert(0, gym_path)

def main():
    print("🚁 PIDTuner演示")
    print("=" * 50)
    
    try:
        # 导入PIDTuner（需要先切换到正确目录）
        scripts_dir = os.path.join(gym_path, 'gym_pybullet_drones', 'scripts')
        os.chdir(scripts_dir)
        
        # 现在导入PIDTuner
        from PIDTuner import PIDTuner
        from gym_pybullet_drones.utils.enums import DroneModel
        
        print("✅ Successfully imported PIDTuner")
        
        # 创建调优器
        tuner = PIDTuner(task_type="hover", drone_model=DroneModel.CF2P)
        
        print(f"📊 Search bounds: {tuner.param_bounds}")
        print(f"⚖️ Optimization weights: {tuner.weights}")
        
        # 运行小规模优化演示
        print("\n🚀 Running quick optimization demo (15 evaluations)...")
        
        result = tuner.optimize_differential_evolution(
            max_evaluations=15,
            population_size=5
        )
        
        if result and tuner.best_params:
            print("\n🎉 Demo completed!")
            print(f"🏆 Best score: {tuner.best_score:.4f}")
            print("📋 Best parameters:")
            for key, value in tuner.best_params.items():
                if key != 'name':
                    print(f"  {key}: {value}")
            
            print(f"\n💾 Parameters saved to Best_PID_Params/best_pid_params.json")
            
        else:
            print("❌ Demo failed")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the correct conda environment (drones)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
