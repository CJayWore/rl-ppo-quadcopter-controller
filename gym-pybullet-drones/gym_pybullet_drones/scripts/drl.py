#!/usr/bin/env python3
"""
Modular Deep Reinforcement Learning Script for Multi-Task Drone Control

This script uses a modular framework approach for training and evaluating 
PPO agents for various drone control tasks.

cd gym-pybullet-drones/gym_pybullet_drones/scripts

Usage Examples:
------------------------------------------------------------------------------------------------------------
    # Disable obstacles
    # Training
    python drl.py --task unified --train_mode True --episodes 1000 --enable_obstacles False
    
    # Evaluation  
    python drl.py --task unified --train_mode False --gui True --enable_obstacles False
    
    # Continue training
    python drl.py --task unified --train_mode True --load_model results/unified/best_model.zip --enable_obstacles False --episodes 10000
------------------------------------------------------------------------------------------------------------
    # Enable obstacles
    # Training
    python drl.py --task unified --train_mode True --episodes 1000
    
    # Evaluation  
    python drl.py --task unified --train_mode False --gui True --duration_sec 180
    
    # Continue training
    python drl.py --task unified --train_mode True --load_model results/unified/best_model.zip --episodes 10000
    
    # List available models
    python drl.py --list_models

-------------------------------------------------------------------------------------------------------------
    # Enable Gaussian noise
    --enable_noise True --noise_level light
    --noise_level light/medium/heavy
    -- noise_decay True/False (default: True)

-------------------------------------------------------------------------------------------------------------
    # Performance Evaluation
    # Evaluate default model (results/unified/best_model.zip) with 100 episodes
    python drl.py --performance_eval --eval_episodes 100
    
    # Evaluate specific model with comprehensive metrics
    python drl.py --performance_eval --evaluate_model results/unified/best_model.zip --eval_episodes 100
    
    # Detailed evaluation with custom duration and LaTeX output
    python drl.py --performance_eval --evaluate_model results/unified/best_model.zip --eval_episodes 50 --eval_duration 180 --generate_latex True
    
    # Evaluation with custom output directory for papers
    python drl.py --performance_eval --evaluate_model results/unified/best_model.zip --output_dir paper_results --generate_latex True
"""

import os
import sys

# Add the framework to the Python path
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
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create configuration objects
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
    
    # Handle list models request
    if args.list_models:
        model_manager = ModelManager(args.output_folder)
        model_manager.list_available_models()
        return
    
    # Handle performance evaluation requests
    if args.performance_eval:
        print("📊 Running Performance Evaluation...")
        
        # Use custom output directory if specified
        output_folder = args.output_dir if args.output_dir else args.output_folder
        logger = DronePerformanceLogger(output_folder)
        
        model_name = args.model_names[0] if args.model_names and len(args.model_names) > 0 else None
        num_episodes = args.eval_episodes
        
        # Evaluate the model
        results = logger.evaluate_model(args.evaluate_model, num_episodes, model_name, args.eval_duration)
        
        if results:
            # Generate reports and visualizations
            logger.generate_performance_report()
            if args.generate_latex:
                logger.generate_latex_table()
            logger.visualize_performance(save_plots=True)
        
        print(f"✅ Performance evaluation completed!")
        return
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Create trainer and run
    trainer = DroneRLTrainer(training_config, env_config, model_config)
    
    # Print configuration summary
    print("🚀 Modular Drone RL Framework")
    print("="*50)
    print(f"🎯 Task: {args.task}")
    if args.task == "trajectory":
        print(f"🛤️  Trajectory type: {args.trajectory_type}")
    print(f"🎮 Mode: {'Training' if args.train_mode else 'Evaluation'}")
    print(f"📁 Output folder: {args.output_folder}")
    if args.task == "unified":
        print(f"🚧 Obstacles: {'Enabled' if args.enable_obstacles else 'Disabled'}")
    print("="*50)
    
    try:
        if args.train_mode:
            # Training mode
            result_folder = trainer.run_training()
            
            if not args.colab:
                print(f"\n🎉 Training completed successfully!")
                print(f"📊 Results saved to: {result_folder}")
                print(f"\n💡 Next steps:")
                print(f"   • List models: python d r l.py --list_models")
                print(f"   • Evaluate: python drl.py --task {args.task} --train_mode False --gui True")
                if args.enable_obstacles and args.task == "unified":
                    print(f"   • Test without obstacles: python drl.py --task {args.task} --train_mode False --enable_obstacles False --gui True")
        else:
            # Evaluation mode
            trainer.run_evaluation(args.load_model)
            print(f"\n🎯 Evaluation completed!")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Training/Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
