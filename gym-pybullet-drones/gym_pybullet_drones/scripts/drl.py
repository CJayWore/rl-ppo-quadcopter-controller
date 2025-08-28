#!/usr/bin/env python3
"""
Modular Deep Reinforcement Learning Script for Multi-Task Drone Control

This script provides a modular framework for training and evaluating 
PPO agents on various drone control tasks including navigation, 
obstacle avoidance, and hovering.

Usage Examples:
    Training:
        python drl.py --task unified --train_mode True --episodes 1000
        python drl.py --task unified --train_mode True --load_model results/unified/best_model.zip --episodes 10000
    
    Evaluation:
        python drl.py --performance_eval --evaluate_model results/unified/best_model.zip --eval_episodes 1000 --gui False --generate_latex True
        mv results/performance_analysis results/performance_PPO_noNoise
        python drl.py --performance_eval --evaluate_model results/unified/best_model.zip --eval_episodes 1000 --gui False --generate_latex True --enable_noise True --noise_level medium
        mv results/performance_analysis results/performance_PPO_withNoise

        python drl.py --performance_eval --evaluate_model results/unified/best_model.zip --eval_episodes 100 --gui False --generate_latex True
    
    With Gaussian Noise:
        python drl.py --task unified --train_mode True --enable_noise True --noise_level medium
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_framework import (
    TrainingConfig, 
    EnvironmentConfig, 
    ModelConfig,
    DroneRLTrainer,
    ModelManager,
    DronePerformanceLogger,
    create_argument_parser
)

def main():
    """Main entry point for the modular RL script."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    training_config = TrainingConfig(
        task=args.task,
        trajectory_type=args.trajectory_type,
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        eval_freq=args.eval_freq,
        enable_obstacles=args.enable_obstacles
    )
    
    env_config = EnvironmentConfig(
        obs_type=args.obs_type,
        act_type=args.act_type,
        gui=args.gui,
        record_video=args.record_video,
        duration_sec=args.duration_sec
    )
    
    model_config = ModelConfig(
        load_model=args.load_model,
        output_folder=args.output_folder,
        colab=args.colab
    )
    
    if args.list_models:
        model_manager = ModelManager(args.output_folder)
        model_manager.list_available_models()
        return
    
    if args.performance_eval:
        print("Running Performance Evaluation...")
        # print(f"GUI Mode: {'Enabled' if args.gui else 'Disabled'}")
        
        output_folder = args.output_dir if args.output_dir else args.output_folder
        logger = DronePerformanceLogger(output_folder)
        
        model_name = args.model_names[0] if args.model_names and len(args.model_names) > 0 else None
        num_episodes = args.eval_episodes
        
        results = logger.evaluate_model(args.evaluate_model, num_episodes, model_name, args.eval_duration, gui_enabled=args.gui)
        
        if results:
            logger.generate_performance_report()
            if args.generate_latex:
                logger.generate_latex_table()
            logger.visualize_performance(save_plots=True)
        
        print("Performance evaluation completed!")
        return
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    trainer = DroneRLTrainer(training_config, env_config, model_config)
    
    print("Modular Drone RL Framework")
    print("="*50)
    print(f"Task: {args.task}")
    if args.task == "trajectory":
        print(f"Trajectory type: {args.trajectory_type}")
    print(f"Mode: {'Training' if args.train_mode else 'Evaluation'}")
    print(f"Output folder: {args.output_folder}")
    if args.task == "unified":
        print(f"Obstacles: {'Enabled' if args.enable_obstacles else 'Disabled'}")
    print("="*50)
    
    try:
        if args.train_mode:
            result_folder = trainer.run_training()
            
            if not args.colab:
                print(f"\nTraining completed successfully!")
                print(f"Results saved to: {result_folder}")
                print(f"\nNext steps:")
                print(f"   • List models: python drl.py --list_models")
                print(f"   • Evaluate: python drl.py --task {args.task} --train_mode False --gui True")
                if args.enable_obstacles and args.task == "unified":
                    print(f"   • Test without obstacles: python drl.py --task {args.task} --train_mode False --enable_obstacles False --gui True")
        else:
            trainer.run_evaluation(args.load_model)
            print(f"\nEvaluation completed!")
            
    except KeyboardInterrupt:
        print(f"\nTraining/Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
