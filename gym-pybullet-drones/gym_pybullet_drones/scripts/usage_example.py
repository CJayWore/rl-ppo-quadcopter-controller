#!/usr/bin/env python3
"""
使用示例：如何使用增强的 DronePerformanceLogger

这个示例展示了如何在实际的模型评估中使用新的性能记录器，
它结合了 Logger.py 的数据采集方式和高级的可视化功能。
"""

import os
import sys

def example_usage():
    """展示如何使用增强的性能记录器"""
    
    print("🚁 DronePerformanceLogger 使用示例")
    print("=" * 50)
    
    # 导入性能记录器
    from rl_framework.performance_evaluation import DronePerformanceLogger
    
    print("\n📋 1. 初始化性能记录器")
    print("-" * 30)
    
    # 创建性能记录器
    # 它继承了 Logger 类的所有数据采集功能，同时添加了高级分析能力
    performance_logger = DronePerformanceLogger(
        output_folder="evaluation_results",  # 输出文件夹
        logging_freq=240,                   # 记录频率 (Hz) - 与 Logger 相同
        num_drones=1,                       # 无人机数量
        duration_sec=180                    # 预分配持续时间（秒）
    )
    
    print(f"✅ 性能记录器初始化完成")
    print(f"   - 继承自 Logger: {hasattr(performance_logger, 'states')}")
    print(f"   - 拥有性能分析功能: {hasattr(performance_logger, 'visualize_performance')}")
    print(f"   - 数据结构与 Logger 兼容: {hasattr(performance_logger, 'log')}")
    
    print("\n📊 2. 数据采集方式对比")
    print("-" * 30)
    
    print("🔹 原 Logger.py 方式:")
    print("   logger.log(drone_id, timestamp, state_20d, control_12d)")
    
    print("🔹 新增强方式:")
    print("   logger.log_step_with_performance(drone_id, timestamp, state_20d, action, reward, info, control_12d)")
    print("   - 保持 Logger 数据结构")  
    print("   - 增加性能指标计算")
    print("   - 增加任务完成检测")
    
    print("\n🎯 3. 关键特性")
    print("-" * 30)
    
    # 显示数据摘要
    summary = performance_logger.get_logger_summary()
    print(f"📈 Logger 数据结构:")
    for key, value in summary.items():
        if key != 'state_labels' and key != 'control_labels':
            print(f"   - {key}: {value}")
    
    print(f"\n🏷️  状态变量 (16维):")
    for i, label in enumerate(summary['state_labels']):
        print(f"   [{i:2d}] {label}")
    
    print(f"\n🎮 控制变量 (12维):")
    for i, label in enumerate(summary['control_labels']):
        print(f"   [{i:2d}] {label}")
    
    print("\n💾 4. 数据保存功能")
    print("-" * 30)
    
    print("🔹 Logger 原生保存:")
    print("   - save(): .npy 格式 (timestamps, states, controls)")
    print("   - save_as_csv(): CSV 文件 (x, y, z, roll, pitch, yaw, etc.)")
    print("   - plot(): 标准 Logger 可视化")
    
    print("🔹 增强保存功能:")
    print("   - save_performance_data(): 所有数据 + 性能指标")
    print("   - visualize_performance(): 高级可视化分析")
    print("   - generate_performance_report(): Markdown 报告")
    print("   - generate_latex_table(): LaTeX 表格")
    
    print("\n🎨 5. 可视化功能")
    print("-" * 30)
    
    print("🔹 Logger 标准图表:")
    print("   - 位置、速度、姿态时间序列")
    print("   - RPM/PWM 时间序列")
    print("   - 10x2 子图布局")
    
    print("🔹 增强性能图表:")
    print("   - 位置误差分析")
    print("   - 任务完成率统计")
    print("   - 控制能量分析") 
    print("   - 安全事件统计")
    print("   - 3D 飞行轨迹")
    print("   - 性能雷达图")
    
    print("\n🚀 6. 使用场景")
    print("-" * 30)
    
    print("✨ 适用于:")
    print("   - 强化学习模型评估")
    print("   - 控制算法性能对比")
    print("   - 飞行数据详细分析")
    print("   - 学术论文数据可视化")
    
    print("\n📝 7. 实际使用步骤")
    print("-" * 30)
    
    print("```python")
    print("# 1. 创建记录器")
    print("logger = DronePerformanceLogger('results', 240, 1, 180)")
    print("")
    print("# 2. 开始记录剧集")
    print("logger.start_episode(drone_id=0)")
    print("")
    print("# 3. 记录每一步 (环境循环中)")
    print("logger.log_step_with_performance(")
    print("    drone=0, timestamp=t, state=state_20d,")
    print("    action=action, reward=reward, info=info")
    print(")")
    print("")
    print("# 4. 结束剧集") 
    print("logger.end_episode(drone_id=0)")
    print("")
    print("# 5. 保存所有数据")
    print("logger.save_performance_data('model_name')")
    print("")
    print("# 6. 生成可视化")
    print("logger.plot_performance_vs_logger()")
    print("```")
    
    print("\n🎯 8. 实际评估模型")
    print("-" * 30)
    
    print("如需评估已训练的模型，使用:")
    print("```python")
    print("results = logger.evaluate_model(")
    print("    'path/to/model.zip',") 
    print("    num_episodes=10,")
    print("    model_name='my_model',")
    print("    duration_sec=180")
    print(")")
    print("```")
    
    print(f"\n{'=' * 50}")
    print("🎉 现在您可以享受 Logger.py 的可靠数据采集")
    print("   加上 performance_evaluation.py 的强大分析!")
    print("=" * 50)

if __name__ == "__main__":
    example_usage()
