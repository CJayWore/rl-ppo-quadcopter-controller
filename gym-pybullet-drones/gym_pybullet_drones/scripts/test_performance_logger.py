#!/usr/bin/env python3
"""
Test script for the enhanced DronePerformanceLogger

This script tests the new performance logger that integrates Logger.py data collection
with advanced visualization capabilities.
"""

import numpy as np
import sys
import os

# Add the rl_framework to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rl_framework'))

from rl_framework.performance_evaluation import DronePerformanceLogger

def test_performance_logger():
    """Test the enhanced performance logger."""
    print("🧪 Testing Enhanced DronePerformanceLogger...")
    
    # Initialize logger
    logger = DronePerformanceLogger(
        output_folder="test_results",
        logging_freq=240,
        num_drones=1,
        duration_sec=10  # Small duration for testing
    )
    
    print(f"✅ Logger initialized successfully!")
    print(f"   Logger type: {type(logger)}")
    print(f"   Inherits from Logger: {hasattr(logger, 'states')}")
    print(f"   Has performance methods: {hasattr(logger, 'visualize_performance')}")
    
    # Test episode recording
    print("\n📝 Testing episode recording...")
    
    # Simulate 2 short episodes
    for episode in range(2):
        print(f"\n   Episode {episode + 1}")
        logger.start_episode(drone_id=0)
        
        # Simulate flight data
        for step in range(50):  # 50 steps per episode
            timestamp = step / 240.0  # 240 Hz
            
            # Create realistic state data (20 elements as per Logger)
            state = np.zeros(20)
            # Position (oscillating around target)
            state[0] = 0.1 * np.sin(step * 0.1) + np.random.normal(0, 0.01)  # x
            state[1] = 0.1 * np.cos(step * 0.1) + np.random.normal(0, 0.01)  # y  
            state[2] = 1.0 + 0.05 * np.sin(step * 0.05) + np.random.normal(0, 0.01)  # z
            
            # Velocity
            state[10:13] = np.random.normal(0, 0.1, 3)  # vel_x, vel_y, vel_z
            
            # Attitude (small angles)
            state[7:10] = np.random.normal(0, 0.05, 3)  # roll, pitch, yaw
            
            # Angular velocity
            state[13:16] = np.random.normal(0, 0.1, 3)  # ang_vel
            
            # RPM (realistic values)
            state[16:20] = 15000 + np.random.normal(0, 500, 4)  # rpm0-3
            
            # Action (4 motors)
            action = np.random.uniform(-0.1, 0.1, 4)
            
            # Reward
            reward = -np.linalg.norm(state[0:3] - np.array([0, 0, 1]))  # Distance penalty
            
            # Info
            info = {
                'target_pos': [0, 0, 1],
                'current_pos': state[0:3].tolist()
            }
            
            # Control (12 elements, zeros for test)
            control = np.zeros(12)
            
            # Log the step
            logger.log_step_with_performance(
                drone=0,
                timestamp=timestamp,
                state=state,
                action=action,
                reward=reward,
                info=info,
                control=control
            )
        
        # End episode
        logger.end_episode(drone_id=0)
    
    # Test data summary
    print("\n📊 Testing data summary...")
    summary = logger.get_logger_summary()
    print(f"   Episodes recorded: {summary['total_episodes']}")
    print(f"   Data shapes: {summary['data_shape']}")
    print(f"   Final counters: {summary['counters']}")
    
    # Test metrics
    print("\n📈 Testing metrics computation...")
    print(f"   Position errors recorded: {len(logger.metrics.position_errors)}")
    print(f"   Task completion rate: {logger.metrics.task_completion_rate:.2%}")
    print(f"   Episodes data: {len(logger.episode_data)}")
    print(f"   Control energy values: {len(logger.metrics.control_energy)}")
    
    # Test saving
    print("\n💾 Testing data saving...")
    try:
        logger.save_performance_data("test")
        print("✅ Data saving successful!")
    except Exception as e:
        print(f"❌ Data saving failed: {e}")
    
    # Test plotting (without showing plots)
    print("\n🎨 Testing plot generation...")
    try:
        # This will generate plots but not show them
        logger.visualize_performance(save_plots=True)
        print("✅ Visualization successful!")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
    
    print("\n🎉 All tests completed!")
    return logger

if __name__ == "__main__":
    test_logger = test_performance_logger()
