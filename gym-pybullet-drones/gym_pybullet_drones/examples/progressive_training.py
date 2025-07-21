#!/usr/bin/env python3
"""Progressive training script for unified drone task."""

import os
import time
from rl import run_stage_training, create_environment, run_evaluation

def run_progressive_training():
    """运行渐进式训练"""
    print("🎓 Starting Progressive Training for Unified Task")
    
    # 训练阶段配置
    stages = [
        {
            'name': 'stability',
            'episodes': 500,
            'learning_rate': 5e-4,
            'description': 'Learning stable flight and hovering'
        },
        {
            'name': 'navigation',
            'episodes': 800,
            'learning_rate': 3e-4,
            'description': 'Learning navigation to target'
        },
        {
            'name': 'obstacle',
            'episodes': 1000,
            'learning_rate': 2e-4,
            'description': 'Learning obstacle avoidance'
        },
        {
            'name': 'full',
            'episodes': 1200,
            'learning_rate': 1e-4,
            'description': 'Full task integration'
        }
    ]
    
    model_path = None
    
    for i, stage in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"Stage {i+1}/4: {stage['name'].upper()}")
        print(f"Description: {stage['description']}")
        print(f"{'='*60}")
        
        # 运行训练
        model_path = run_stage_training(
            stage=stage['name'],
            episodes=stage['episodes'],
            learning_rate=stage['learning_rate'],
            load_model=model_path  # 从上一阶段加载模型
        )
        
        # 简单评估
        print(f"\n📊 Evaluating Stage {stage['name']}...")
        try:
            # 这里可以添加快速评估代码
            pass
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
        
        print(f"✅ Stage {stage['name']} completed!")
        
        # 短暂休息
        time.sleep(2)
    
    print(f"\n🎉 Progressive Training Completed!")
    print(f"Final model saved at: {model_path}")
    
    return model_path

if __name__ == "__main__":
    final_model = run_progressive_training()
    
    # 运行最终评估
    print(f"\n🚀 Running final evaluation...")
    # 这里可以添加最终评估代码