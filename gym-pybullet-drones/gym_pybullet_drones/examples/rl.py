"""Script demonstrating Deep Reinforcement Learning control for multiple tasks.

This script supports training and evaluation of PPO agents for:
- Hover task: Keep drone at a fixed position
- Trajectory task: Follow predefined paths (circle, figure8, waypoints)
- Obstacle task: Navigate through obstacles to reach a goal

How to use
---------------------------------------------------------------------------------------------------------
# 无障碍物训练 - 专注于导航和悬停
python rl.py --task unified --train_mode True --enable_obstacles False --episodes 1000 --gui False

# 长时间无障碍物训练
python rl.py --task unified --train_mode True --enable_obstacles False --episodes 2000 --learning_rate 3e-4 --gui False

# 标准有障碍物训练
python rl.py --task unified --train_mode True --enable_obstacles True --episodes 1000 --gui False

# 用已有模型继续训练
python rl.py --task unified --train_mode True --enable_obstacles True --load_model results/unified/best_model.zip --episodes 500 --gui False

# 在无障碍物环境中评估
python rl.py --task unified --train_mode False --enable_obstacles False --gui True --duration_sec 60

# 在有障碍物环境中评估
python rl.py --task unified --train_mode False --enable_obstacles True --gui True --duration_sec 60
---------------------------------------------------------------------------------------------------------
5. Continue training from a saved model:
    # 继续训练悬停任务
    python rl.py --task hover --train_mode True --load_model results/hover/best_model.zip --episodes 500 --gui False

    # 继续训练轨迹跟踪
    python rl.py --task trajectory --trajectory_type circle --train_mode True --load_model results/trajectory_circle/best_model.zip --episodes 500 --gui False

    # 继续训练避障任务
    python rl.py --task unified --train_mode True --load_model results/unified/best_model.zip --episodes 500 --gui False

---------------------------------------------------------------------------------------------------------

Notes
-----
This integrates `gym-pybullet-drones` with `stable-baselines3` for multiple RL tasks.
"""
import os
import time
# import json
from datetime import datetime
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.TrajectoryAviary import TrajectoryAviary
from gym_pybullet_drones.envs.ObstacleAviary import ObstacleAviary
from gym_pybullet_drones.envs.UnifiedAviary import UnifiedAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Default parameters
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')
DEFAULT_TASK = "hover"
DEFAULT_TRAJECTORY_TYPE = "circle"
DEFAULT_TRAIN_MODE = True
DEFAULT_EPISODES = 1000
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_EVAL_FREQ = 2000
DEFAULT_DURATION_SEC = 30

def get_target_reward(task, act_type):
    """Get target reward based on task and action type."""
    if task == "hover":
        return 474.15 if act_type == ActionType.ONE_D_RPM else 100000.0
    elif task == "trajectory":
        return 400.0
    elif task == "obstacle":
        return 300.0
    elif task == "unified":  
        return 30000.0
    else:
        return 300.0

def create_environment(task, trajectory_type="circle", **kwargs):
    """Create environment based on task type."""
    if task == "hover":
        return HoverAviary(**kwargs)
    elif task == "trajectory":
        return TrajectoryAviary(trajectory_type=trajectory_type, **kwargs)
    elif task == "obstacle":
        return ObstacleAviary(**kwargs)
    elif task == "unified":  # 新增统一任务
        return UnifiedAviary(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")

def run_training(task, trajectory_type, output_folder, episodes, learning_rate, eval_freq, 
                load_model, gui, record_video, obs_type, act_type, colab, enable_obstacles=True):
    """Run training for specified task."""
    
    # Create task-specific output folder (without timestamp)
    task_folder = f"{task}_{trajectory_type}" if task == "trajectory" else task
    output_path = os.path.join(output_folder, task_folder)
    
    # Create timestamped subfolder for this training session
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    session_folder = os.path.join(output_path, f"session_{timestamp}")
    
    # Create directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(session_folder, exist_ok=True)
    
    print(f"   Starting training for {task} task...")
    print(f"   Task folder: {output_path}")
    print(f"   Session folder: {session_folder}")
    
    # Create training environment
    env_kwargs = dict(obs=obs_type, act=act_type, gui=False, record=False)
    
    if task == "trajectory":
        env_kwargs.update({
            'trajectory_type': trajectory_type,
            'trajectory_radius': 0.8,
            'trajectory_height': 1.0,
            'max_distance_from_path': 2.0
        })
    elif task == "obstacle":
        env_kwargs.update({
            'num_obstacles': 5,
            'obstacle_radius': 0.3,
            'sensing_range': 2.0
        })
    elif task == "unified":
        env_kwargs.update({
            'num_obstacles': 8 if enable_obstacles else 0,
            'obstacle_radius': 0.25,
            'sensing_range': 2.0,
            'target_radius': 0.15,
            # 'hover_threshold': 1.0,
            'episode_len_sec': 30,
            'randomize_init': True,
            'enable_obstacles': enable_obstacles
        })
    
    # Create vectorized training environment
    train_env = make_vec_env(
        lambda: create_environment(task, trajectory_type, **env_kwargs),
        n_envs=1,
        seed=0
    )
    
    # Create evaluation environment
    eval_env = create_environment(task, trajectory_type, **env_kwargs)
    
    # Check environment spaces
    print(f'[INFO] Action space: {train_env.action_space}')
    print(f'[INFO] Observation space: {train_env.observation_space}')
    
    # Configure network architecture based on task
    if task == "obstacle" or task == "unified":
        policy_kwargs = dict(
            net_arch=[dict(pi=[512, 512, 256], vf=[512, 512, 256])]
            )
    else:
        policy_kwargs = dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]
            )
    
    # Create or load model
    if load_model and os.path.exists(load_model):
        print(f"📥 Loading existing model: {load_model}")
        
        try:
            # Load the existing model
            model = PPO.load(load_model)
            print(f"✅ Model loaded successfully!")
            
            # Check if environments are compatible
            if (model.observation_space != train_env.observation_space or 
                model.action_space != train_env.action_space):
                print("⚠️  Environment spaces don't match! Creating new model with loaded weights...")
                
                # Create new model with current environment
                new_model = PPO(
                    'MlpPolicy',
                    train_env,
                    learning_rate=learning_rate,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    policy_kwargs=policy_kwargs,
                    verbose=1
                )
                
                # Try to transfer compatible weights
                try:
                    new_model.policy.load_state_dict(model.policy.state_dict(), strict=False)
                    print("✅ Transferred compatible weights from old model")
                except Exception as e:
                    print(f"⚠️  Could not transfer weights: {e}")
                    print("   Starting with new random weights")
                
                model = new_model
            else:
                # Set the environment for the loaded model
                model.set_env(train_env)
                print("✅ Environment set for loaded model")

            # Update learning rate (create new optimizer with new learning rate)
            if model.lr_schedule(1.0) != learning_rate:
                print(f"   Updating learning rate from {model.lr_schedule(1.0)} to {learning_rate}")
                
                # Create new model with updated learning rate but keep the trained weights
                updated_model = PPO(
                    'MlpPolicy',
                    train_env,
                    learning_rate=learning_rate,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    policy_kwargs=policy_kwargs,
                    verbose=1
                )
                
                # Copy trained weights
                updated_model.policy.load_state_dict(model.policy.state_dict())
                if hasattr(model, 'value_function'):
                    updated_model.value_function.load_state_dict(model.value_function.state_dict())
                
                model = updated_model
                print("✅ Learning rate updated successfully")
            
            # Print model info
            print(f"   Loaded model info:")
            print(f"   Learning rate: {model.lr_schedule(1.0)}")
            print(f"   Observation space: {model.observation_space}")
            print(f"   Action space: {model.action_space}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Creating new model instead...")
            load_model = None  # Fall back to creating new model
    
    # Create new model if no model to load or loading failed
    if not load_model or not os.path.exists(load_model):
        print("🎯 Creating new PPO model...")
        model = PPO(
            'MlpPolicy',
            train_env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            verbose=1
        )
        
        print(f"   New model info:")
        print(f"   Learning rate: {model.lr_schedule(1.0)}")
        print(f"   Observation space: {model.observation_space}")
        print(f"   Action space: {model.action_space}")
    
    # Set up callbacks
    target_reward = get_target_reward(task, act_type)
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=target_reward,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=session_folder + '/',
        log_path=session_folder + '/',
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    # Train the model
    total_timesteps = episodes * 1000
    print(f"🎮 Training for {total_timesteps} timesteps...")
    
    if load_model and os.path.exists(load_model):
        print(f"   Continuing training from loaded model...")
    else:
        print(f"   Starting training from scratch...")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=100,
        reset_num_timesteps=False if load_model else True  # Don't reset timesteps if continuing
    )
    
    # 训练结束后，访问环境的计时统计
    if hasattr(train_env.envs[0], 'print_final_stats'):
        print("\n" + "="*50)
        print("📈 TRAINING COMPLETED - FINAL STATISTICS")
        train_env.envs[0].print_final_stats()
    
    # 或者获取详细统计数据
    if hasattr(train_env.envs[0], 'get_timing_stats'):
        stats = train_env.envs[0].get_timing_stats()
        
        # 保存统计信息到文件
        import json
        stats_file = os.path.join(output_folder, 'timing_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"📊 Timing statistics saved to: {stats_file}")

    # Close environments
    train_env.close()
    eval_env.close()
    
    # Only save best model to main folder
    session_best_path = os.path.join(session_folder, 'best_model.zip')
    main_best_path = os.path.join(output_path, 'best_model.zip')
    
    if os.path.exists(session_best_path):
        # Copy best model to main folder
        import shutil
        shutil.copy2(session_best_path, main_best_path)
        print(f"   Best model saved to: {main_best_path}")
    else:
        # If no best model found, save current model as best
        print("    No best model found, saving current model as best...")
        model.save(main_best_path)
        print(f"   Current model saved as best: {main_best_path}")
    
    # Save training info
    training_info = {
        'task': task,
        'trajectory_type': trajectory_type if task == "trajectory" else None,
        'episodes': episodes,
        'learning_rate': learning_rate,
        'target_reward': target_reward,
        'timestamp': timestamp,
        'session_folder': session_folder,
        'total_timesteps': total_timesteps,
        'enable_obstacles': enable_obstacles if task == "unified" else None
    }
    
    import json
    with open(os.path.join(output_path, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    # Plot training progress
    eval_file = session_folder + '/evaluations.npz'
    if os.path.exists(eval_file):
        print("\n   Training progression:")
        with np.load(eval_file) as data:
            timesteps = data['timesteps']
            results = data['results']
            
            # Plot training progress
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps, results.mean(axis=1), 'b-', label='Mean Reward')
            plt.fill_between(timesteps, 
                           results.mean(axis=1) - results.std(axis=1),
                           results.mean(axis=1) + results.std(axis=1),
                           alpha=0.2, color='blue')
            plt.axhline(y=target_reward, color='r', linestyle='--', label=f'Target ({target_reward})')
            plt.xlabel('Timesteps')
            plt.ylabel('Mean Reward')
            plt.title(f'Training Progress - {task.capitalize()} Task')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plt.savefig(session_folder + '/training_progress.png', dpi=150, bbox_inches='tight')
            plt.savefig(output_path + '/latest_training_progress.png', dpi=150, bbox_inches='tight')
            
            if not colab:
                plt.show()
            
            # Print final statistics
            if len(results) > 0:
                print(f"   Final performance: {results[-1][0]:.2f} reward")
                print(f"   Target reward: {target_reward}")
                print(f"{'✅ Target reached!' if results[-1][0] >= target_reward else '❌ Target not reached'}")
    
    print(f"\n✅ Training completed!")
    print(f"   Best model saved to: {main_best_path}")
    print(f"   Session details: {session_folder}")
    
    return output_path

def get_model_path(task, trajectory_type=None, output_folder=DEFAULT_OUTPUT_FOLDER):
    """Get the best model path for a given task."""
    task_folder = f"{task}_{trajectory_type}" if task == "trajectory" else task
    output_path = os.path.join(output_folder, task_folder)
    return os.path.join(output_path, 'best_model.zip')

def list_available_models(output_folder=DEFAULT_OUTPUT_FOLDER):
    """List all available trained models."""
    if not os.path.exists(output_folder):
        print(f"❌ Output folder not found: {output_folder}")
        return
    
    print(f"   Available models in {output_folder}:")
    
    tasks = ['hover', 'trajectory_circle', 'trajectory_figure8', 'trajectory_waypoints', 'obstacle','unified']
    found_models = False
    
    for task in tasks:
        task_path = os.path.join(output_folder, task)
        if os.path.exists(task_path):
            best_model = os.path.join(task_path, 'best_model.zip')
            if os.path.exists(best_model):
                found_models = True
                print(f"\n✅ {task.replace('_', ' ').title()}")
                print(f"      Model: {best_model}")
                
                # Check training info
                info_file = os.path.join(task_path, 'training_info.json')
                if os.path.exists(info_file):
                    try:
                        import json  # 移到这里，或者在文件开头导入
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                        print(f"      Last training: {info.get('timestamp', 'Unknown')}")
                        print(f"      Episodes: {info.get('episodes', 'Unknown')}")
                        print(f"      Target reward: {info.get('target_reward', 'Unknown')}")
                    except:
                        pass
                
                # Count session folders
                sessions = [d for d in os.listdir(task_path) if d.startswith('session_')]
                if sessions:
                    print(f"   📁 Training sessions: {len(sessions)}")
    
    if not found_models:
        print("❌ No trained models found.")
        print("💡 Train a model first with: python rl.py --task hover --train_mode True")

def run_evaluation(task, trajectory_type, model_path, output_folder, duration_sec, 
                  gui, record_video, obs_type, act_type, colab, enable_obstacles=True):
    """Run evaluation of trained model."""
    
    print(f"   Starting evaluation for {task} task...")
    print(f"   Model path: {model_path}")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    model = PPO.load(model_path)
    print(f"✅ Model loaded successfully!")
    
    # Create test environments
    env_kwargs = dict(
        obs=obs_type, 
        act=act_type, 
        gui=gui, 
        record=record_video, 
        randomize_init=True
        )
    
    if task == "trajectory":
        env_kwargs.update({
            'trajectory_type': trajectory_type,
            'trajectory_radius': 0.8,
            'trajectory_height': 1.0,
            'max_distance_from_path': 2.0
        })
    elif task == "obstacle":
        env_kwargs.update({
            'num_obstacles': 5,
            'obstacle_radius': 0.3,
            'sensing_range': 2.0
        })
    elif task == "unified":
        env_kwargs.update({
            'num_obstacles': 8 if enable_obstacles else 0,
            'obstacle_radius': 0.25,
            'sensing_range': 2.0,
            'target_radius': 0.15,
            # 'hover_threshold': 1.0,
            'episode_len_sec': 30,
            'randomize_init': True,
            'enable_obstacles': enable_obstacles
        })
    
    print(f"🔧 Environment parameters: {env_kwargs}")

    test_env = create_environment(task, trajectory_type, **env_kwargs)
    
    # Create no-GUI environment for evaluation
    env_kwargs_nogui = env_kwargs.copy()
    env_kwargs_nogui.update({'gui': False, 'record': False})
    test_env_nogui = create_environment(task, trajectory_type, **env_kwargs_nogui)
    
    # Evaluate policy
    print("   Evaluating policy performance...")
    mean_reward, std_reward = evaluate_policy(
        model, test_env_nogui, n_eval_episodes=10
    )
    print(f"   Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Create logger
    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=output_folder,
        colab=colab
    )
    total_reward = 0
    step_count = 0
    episode_count = 0
    # Run visualization
    print(f"Running visualization for {duration_sec} seconds...")
    obs, info = test_env.reset(seed=int(time.time()), options={})
    episode_count += 1
    print(f"Starting evaluation episode {episode_count}...")
    print(f"        Target position: [[{test_env.TARGET_POS[0]:.3f}, {test_env.TARGET_POS[1]:.3f}, {test_env.TARGET_POS[2]:.3f}]")
    initial_state = test_env._getDroneStateVector(0)
    initial_pos = initial_state[0:3]
    print(f"        Initial position: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}]")
    
    start = time.time()
    
    
    
    for i in range(int(duration_sec * test_env.CTRL_FREQ)):
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        step_count += 1
        
        # Log data
        obs_flat = obs.squeeze() if hasattr(obs, 'squeeze') else obs
        act_flat = action.squeeze() if hasattr(action, 'squeeze') else action
        
        if obs_type == ObservationType.KIN:
            # Extract state information
            if len(obs_flat) >= 12:
                state = np.hstack([
                    obs_flat[0:3],      # position
                    np.zeros(4),        # quaternion (placeholder)
                    obs_flat[3:15] if len(obs_flat) >= 15 else np.pad(obs_flat[3:], (0, max(0, 12-len(obs_flat[3:])))),  # velocity and angular velocity
                    act_flat if hasattr(act_flat, '__len__') else [act_flat]
                ])
            else:
                state = np.pad(obs_flat, (0, max(0, 16-len(obs_flat))))
                
            logger.log(
                drone=0,
                timestamp=i / test_env.CTRL_FREQ,
                state=state,
                control=np.zeros(12)
            )
        
        # Print progress
        if i % (test_env.CTRL_FREQ * 2) == 0:
            avg_reward = total_reward / max(step_count, 1)
            print(f"    Time: {i/test_env.CTRL_FREQ:.1f}s, "
                  f"Reward: {reward:.3f}, "
                  f"Avg Reward: {avg_reward:.3f}")
            
            # Task-specific info
            if hasattr(info, 'get'):
                if task == "unified":
                    distance = info.get('distance_to_target', 0)
                    hover_time = info.get('time_at_target', 0)
                    print(f"      Distance to target: {distance:.3f}m, Hover time: {hover_time:.1f}s")
                elif task == "trajectory":
                    progress = info.get('trajectory_progress', 0)
                    print(f"      Progress: {progress:.1%}")
                elif task == "obstacle":
                    distance = info.get('distance_to_goal', 0)
                    print(f"      Distance to goal: {distance:.2f}m")
        
        # Render
        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        
        # Handle episode termination
        if terminated or truncated:
            episode_count += 1
            print(f"Episode {episode_count} ended!")
            print(f"   Terminated: {terminated}, Truncated: {truncated}")
            obs, info = test_env.reset(seed=int(time.time()), options={})

            if task == "unified":
                hover_time = getattr(test_env, 'time_at_target', 0)
                required_time = getattr(test_env, 'required_hover_time', 3.0)
                if terminated:
                    print(f"   ✅ Task completed! Hovered for {hover_time:.1f}s (required: {required_time:.1f}s)")
                else:
                    print(f"   ❌ Task not completed. Hover time: {hover_time:.1f}s (required: {required_time:.1f}s)")
            

    test_env.close()
    test_env_nogui.close() 
    
    # Save and plot results
    print(f"Saving results to: {output_folder}")
    logger.save()
    logger.save_as_csv(f"{task}_evaluation")
    
    if obs_type == ObservationType.KIN:
        logger.plot()
    
    # Print final statistics
    print(f"\nFinal Results:")
    print(f"   Total episodes: {episode_count}")
    print(f"   Total steps: {step_count}")
    print(f"   Average reward per step: {total_reward/max(step_count, 1):.3f}")
    print(f"   Mean evaluation reward: {mean_reward:.2f} ± {std_reward:.2f}")

def run(task=DEFAULT_TASK, trajectory_type=DEFAULT_TRAJECTORY_TYPE, train_mode=DEFAULT_TRAIN_MODE,
        episodes=DEFAULT_EPISODES, learning_rate=DEFAULT_LEARNING_RATE, eval_freq=DEFAULT_EVAL_FREQ,
        load_model=None, output_folder=DEFAULT_OUTPUT_FOLDER, duration_sec=DEFAULT_DURATION_SEC,
        gui=DEFAULT_GUI, record_video=DEFAULT_RECORD_VIDEO, obs_type=DEFAULT_OBS, 
        act_type=DEFAULT_ACT, colab=DEFAULT_COLAB, list_models=False, enable_obstacles=True):
    """Main function to run training or evaluation."""
    
    # List available models if requested
    if list_models:
        list_available_models(output_folder)
        return
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"🎯 Task: {task}")
    if task == "trajectory":
        print(f"🛤️  Trajectory type: {trajectory_type}")
    print(f"🎮 Mode: {'Training' if train_mode else 'Evaluation'}")
    print(f"📁 Output folder: {output_folder}")
    
    if train_mode:
        # Training mode
        result_folder = run_training(
            task=task,
            trajectory_type=trajectory_type,
            output_folder=output_folder,
            episodes=episodes,
            learning_rate=learning_rate,
            eval_freq=eval_freq,
            load_model=load_model,
            gui=gui,
            record_video=record_video,
            obs_type=obs_type,
            act_type=act_type,
            colab=colab,
            enable_obstacles=enable_obstacles
        )
        
        if not colab:
            print(f"\nTraining completed!")
            print(f"To list available models: python rl.py --list_models")
            print(f"To evaluate: python rl.py --task {task} --train_mode False --gui True")
    
    else:
        # Evaluation mode
        if load_model:
            model_path = load_model
        else:
            # Auto-detect model path
            model_path = get_model_path(task, trajectory_type, output_folder)
            
            if not os.path.exists(model_path):
                print(f"❌ No trained model found for {task} task")
                print(f"Available models:")
                list_available_models(output_folder)
                return
            
            print(f"🔍 Using model: {model_path}")
        
        run_evaluation(
            task=task,
            trajectory_type=trajectory_type,
            model_path=model_path,
            output_folder=output_folder,
            duration_sec=duration_sec,
            gui=gui,
            record_video=record_video,
            obs_type=obs_type,
            act_type=act_type,
            colab=colab,
            enable_obstacles=enable_obstacles
        )

if __name__ == '__main__':
    # Define and parse arguments
    parser = argparse.ArgumentParser(description='Deep Reinforcement Learning for Drone Control')
    
    # Task parameters
    parser.add_argument('--task', default=DEFAULT_TASK, type=str,
                   choices=['hover', 'trajectory', 'obstacle', 'unified'],  # 添加 unified
                   help='Task to train/evaluate (default: hover)')
    parser.add_argument('--trajectory_type', default=DEFAULT_TRAJECTORY_TYPE, type=str,
                       choices=['circle', 'figure8', 'waypoints'],
                       help='Trajectory type for trajectory task (default: circle)')
    
    # Training parameters
    parser.add_argument('--train_mode', default=DEFAULT_TRAIN_MODE, type=str2bool,
                       help='Whether to train (True) or evaluate (False) (default: True)')
    parser.add_argument('--episodes', default=DEFAULT_EPISODES, type=int,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--learning_rate', default=DEFAULT_LEARNING_RATE, type=float,
                       help='Learning rate for training (default: 3e-4)')
    parser.add_argument('--eval_freq', default=DEFAULT_EVAL_FREQ, type=int,
                       help='Evaluation frequency during training (default: 2000)')
    
    # Model parameters
    parser.add_argument('--load_model', default=None, type=str,
                       help='Path to model to load (for continuing training or evaluation)')
    
    # Environment parameters
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                       help='Output folder for logs and models (default: results)')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int,
                       help='Duration for evaluation in seconds (default: 30)')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool,
                       help='Whether to use PyBullet GUI (default: True)')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool,
                       help='Whether to record video (default: False)')
    parser.add_argument('--obs_type', default=DEFAULT_OBS, type=ObservationType,
                       help='Observation type (default: kin)')
    parser.add_argument('--act_type', default=DEFAULT_ACT, type=ActionType,
                       help='Action type (default: rpm)')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=str2bool,
                       help='Whether running in Colab (default: False)')
    parser.add_argument('--list_models', action='store_true',
                    help='List all available trained models')
    parser.add_argument('--enable_obstacles', default=True, type=str2bool,
                       help='Whether to enable obstacles for unified task (default: True)')
    

    
    
    ARGS = parser.parse_args()
    
    # Run the main function
    run(**vars(ARGS))