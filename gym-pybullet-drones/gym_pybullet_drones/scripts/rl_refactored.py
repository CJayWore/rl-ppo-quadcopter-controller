"""
Deep Reinforcement Learning Script for Multi-Task Drone Control

This refactored script provides a clean, modular approach to training and evaluating 
PPO agents for various drone control tasks including hover, trajectory following, 
obstacle avoidance, and unified tasks.

Usage Examples:
    # Training
    python rl_refactored.py --task unified --train_mode True --episodes 1000
    
    # Evaluation  
    python rl_refactored.py --task unified --train_mode False --gui True
    
    # Continue training
    python rl_refactored.py --task unified --train_mode True --load_model results/unified/best_model.zip
    
    # 训练（有障碍物）
    python rl_modular.py --train_mode True --episodes 1000

    # 训练（无障碍物）
    python rl_modular.py --train_mode True --enable_obstacles False --episodes 1000

    # 评估（有障碍物）
    python rl_modular.py --train_mode False --gui True

    # 评估（无障碍物）
    python rl_modular.py --train_mode False --enable_obstacles False --gui True
"""

import os
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import subprocess
import platform

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.DRLAviary import UnifiedAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    task: str = "hover"
    trajectory_type: str = "circle"
    episodes: int = 1000
    learning_rate: float = 3e-4
    eval_freq: int = 2000
    enable_obstacles: bool = True
    

@dataclass
class EnvironmentConfig:
    """Environment configuration parameters."""
    obs_type: ObservationType = ObservationType('kin')
    act_type: ActionType = ActionType('rpm')
    gui: bool = True
    record_video: bool = False
    duration_sec: int = 30
    

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    load_model: Optional[str] = None
    output_folder: str = 'results'
    colab: bool = False


class NotificationManager:
    """Handles system notifications for training completion."""
    
    @staticmethod
    def send_system_notification(title: str, message: str, urgent: bool = False) -> bool:
        """Send system notification (supports macOS and cross-device sync)."""
        try:
            if platform.system() == "Darwin":  # macOS
                sound_name = "Glass" if urgent else "Blow"
                
                applescript = f'''
                display notification "{message}" ¬
                    with title "{title}" ¬
                    subtitle "训练状态更新" ¬
                    sound name "{sound_name}"
                '''
                
                subprocess.run(['osascript', '-e', applescript], check=True)
                print("✅ macOS 系统通知发送成功！")
                print("📱 如果设置正确，iPhone 应该也会收到通知")
                return True
            else:
                print("⚠️  当前系统不支持 macOS 通知")
                return False
        except Exception as e:
            print(f"❌ 系统通知发送失败: {e}")
            return False
    
    @classmethod
    def send_training_completion_notification(
        cls, 
        task: str, 
        episodes: int, 
        final_reward: str, 
        target_reward: float, 
        training_time: float,
        enable_obstacles: Optional[bool] = None
    ) -> bool:
        """Send training completion notification."""
        if isinstance(final_reward, str) or final_reward == "未知":
            success = False
        else:
            try:
                success = float(final_reward) >= target_reward
            except (ValueError, TypeError):
                success = False
                
        status_emoji = "🎉" if success else "⚠️"
        title = f"{status_emoji} 无人机训练完成"
        
        obstacle_info = ""
        if task == "unified" and enable_obstacles is not None:
            obstacle_info = f"\\n障碍物: {'启用' if enable_obstacles else '禁用'}"
        
        message = (f"任务: {task.upper()}\\n轮数: {episodes}\\n"
                  f"性能: {final_reward}/{target_reward}\\n"
                  f"用时: {training_time/3600:.1f}h{obstacle_info}")
        
        return cls.send_system_notification(title, message, urgent=success)


class EnvironmentFactory:
    """Factory for creating different types of environments."""
    
    TARGET_REWARDS = {
        "hover": {ActionType.ONE_D_RPM: 474.15, "default": 100000.0},
        "trajectory": {"default": 400.0},
        "obstacle": {"default": 300.0},
        "unified": {"default": 500000.0}
    }
    
    NETWORK_CONFIGS = {
        "complex": dict(net_arch=[dict(pi=[512, 512, 256], vf=[512, 512, 256])]),
        "simple": dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    }
    
    @classmethod
    def get_target_reward(cls, task: str, act_type: ActionType) -> float:
        """Get target reward based on task and action type."""
        task_rewards = cls.TARGET_REWARDS.get(task, {"default": 300.0})
        return task_rewards.get(act_type, task_rewards["default"])
    
    @classmethod
    def create_environment(cls, task: str, trajectory_type: str = "circle", **kwargs):
        """Create environment based on task type."""
        env_map = {
            "unified": UnifiedAviary
        }
        
        if task not in env_map:
            raise ValueError(f"Unknown task: {task}")
            
        return env_map[task](**kwargs)
    
    @classmethod
    def get_env_kwargs(cls, task: str, config: EnvironmentConfig, training_config: TrainingConfig) -> Dict[str, Any]:
        """Get environment-specific keyword arguments."""
        base_kwargs = {
            'obs': config.obs_type,
            'act': config.act_type,
            'gui': False,  # Training always uses False
            'record': False
        }
        
        task_specific = {
            "trajectory": {
                'trajectory_type': training_config.trajectory_type,
                'trajectory_radius': 0.8,
                'trajectory_height': 1.0,
                'max_distance_from_path': 2.0
            },
            "obstacle": {
                'num_obstacles': 5,
                'obstacle_radius': 0.3,
                'sensing_range': 2.0
            },
            "unified": {
                'num_obstacles': 8 if training_config.enable_obstacles else 0,
                'obstacle_radius': 0.25,
                'sensing_range': 2.0,
                'target_radius': 0.15,
                'episode_len_sec': 30,
                'randomize_init': True,
                'enable_obstacles': training_config.enable_obstacles
            }
        }
        
        base_kwargs.update(task_specific.get(task, {}))
        return base_kwargs
    
    @classmethod
    def get_network_config(cls, task: str) -> Dict[str, Any]:
        """Get network configuration based on task complexity."""
        complex_tasks = ["obstacle", "unified"]
        config_type = "complex" if task in complex_tasks else "simple"
        return cls.NETWORK_CONFIGS[config_type]


class ModelManager:
    """Manages model loading, saving, and configuration."""
    
    def __init__(self, output_folder: str):
        self.output_folder = output_folder
    
    def get_model_path(self, task: str, trajectory_type: Optional[str] = None) -> str:
        """Get the best model path for a given task."""
        task_folder = f"{task}_{trajectory_type}" if task == "trajectory" else task
        output_path = os.path.join(self.output_folder, task_folder)
        return os.path.join(output_path, 'best_model.zip')
    
    def get_stats_path(self, task: str, trajectory_type: Optional[str] = None) -> str:
        """Get the VecNormalize stats path for a given task."""
        task_folder = f"{task}_{trajectory_type}" if task == "trajectory" else task
        output_path = os.path.join(self.output_folder, task_folder)
        return os.path.join(output_path, "vec_normalize.pkl")
    
    def create_ppo_model(self, env, learning_rate: float, task: str) -> PPO:
        """Create a new PPO model with appropriate configuration."""
        policy_kwargs = EnvironmentFactory.get_network_config(task)
        
        return PPO(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device='auto',
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5
        )
    
    def load_model_with_compatibility_check(
        self, 
        model_path: str, 
        env, 
        learning_rate: float, 
        task: str
    ) -> PPO:
        """Load model with compatibility checking and updating."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"📥 Loading existing model: {model_path}")
        
        try:
            model = PPO.load(model_path)
            print("✅ Model loaded successfully!")
            
            # Check environment compatibility
            if (model.observation_space != env.observation_space or 
                model.action_space != env.action_space):
                print("⚠️  Environment spaces don't match! Creating new model with loaded weights...")
                model = self._transfer_weights_to_new_model(model, env, learning_rate, task)
            else:
                model.set_env(env)
                print("✅ Environment set for loaded model")
            
            # Update learning rate if needed
            if model.lr_schedule(1.0) != learning_rate:
                model = self._update_learning_rate(model, env, learning_rate, task)
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Creating new model instead...")
            return self.create_ppo_model(env, learning_rate, task)
    
    def _transfer_weights_to_new_model(self, old_model, env, learning_rate: float, task: str) -> PPO:
        """Transfer compatible weights from old model to new model."""
        new_model = self.create_ppo_model(env, learning_rate, task)
        
        try:
            new_model.policy.load_state_dict(old_model.policy.state_dict(), strict=False)
            print("✅ Transferred compatible weights from old model")
        except Exception as e:
            print(f"⚠️  Could not transfer weights: {e}")
            print("   Starting with new random weights")
        
        return new_model
    
    def _update_learning_rate(self, model, env, learning_rate: float, task: str) -> PPO:
        """Update model learning rate by creating new model with same weights."""
        print(f"   Updating learning rate from {model.lr_schedule(1.0)} to {learning_rate}")
        
        updated_model = self.create_ppo_model(env, learning_rate, task)
        updated_model.policy.load_state_dict(model.policy.state_dict())
        
        if hasattr(model, 'value_function'):
            updated_model.value_function.load_state_dict(model.value_function.state_dict())
        
        print("✅ Learning rate updated successfully")
        return updated_model
    
    def list_available_models(self):
        """List all available trained models."""
        if not os.path.exists(self.output_folder):
            print(f"❌ Output folder not found: {self.output_folder}")
            return
        
        print(f"   Available models in {self.output_folder}:")
        
        tasks = ['hover', 'trajectory_circle', 'trajectory_figure8', 'trajectory_waypoints', 'obstacle', 'unified']
        found_models = False
        
        for task in tasks:
            task_path = os.path.join(self.output_folder, task)
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
            print("💡 Train a model first with: python rl_refactored.py --task hover --train_mode True")


class DroneRLTrainer:
    """Main class for drone reinforcement learning training and evaluation."""
    
    def __init__(
        self, 
        training_config: TrainingConfig,
        env_config: EnvironmentConfig,
        model_config: ModelConfig
    ):
        self.training_config = training_config
        self.env_config = env_config
        self.model_config = model_config
        self.model_manager = ModelManager(model_config.output_folder)
        self.notification_manager = NotificationManager()
    
    def run_training(self) -> str:
        """Run training for the specified task."""
        training_start_time = time.time()
        
        # Setup output directories
        output_path, session_folder = self._setup_output_directories()
        
        print(f"   Starting training for {self.training_config.task} task...")
        print(f"   Task folder: {output_path}")
        print(f"   Session folder: {session_folder}")
        
        # Create environments
        train_env, eval_env = self._create_training_environments(output_path)
        
        # Create or load model
        model = self._setup_model(train_env, output_path)
        
        # Setup callbacks and train
        eval_callback = self._setup_callbacks(eval_env, session_folder)
        self._train_model(model, eval_callback)
        
        # Post-training cleanup and notifications
        self._post_training_cleanup(train_env, eval_env, output_path, session_folder, training_start_time)
        
        return output_path
    
    def run_evaluation(self, model_path: Optional[str] = None):
        """Run evaluation of trained model."""
        print(f"   Starting evaluation for {self.training_config.task} task...")
        
        # Determine model path
        if model_path is None:
            model_path = self.model_manager.get_model_path(
                self.training_config.task, 
                self.training_config.trajectory_type
            )
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("Available models:")
            self.model_manager.list_available_models()
            return
        
        print(f"   Model path: {model_path}")
        
        # Load model and create evaluation environment
        model, test_env, test_env_nogui = self._setup_evaluation_environment(model_path)
        
        # Run evaluation
        self._perform_evaluation(model, test_env, test_env_nogui)
    
    def _setup_output_directories(self) -> Tuple[str, str]:
        """Setup output directories for training."""
        task_folder = (f"{self.training_config.task}_{self.training_config.trajectory_type}" 
                      if self.training_config.task == "trajectory" 
                      else self.training_config.task)
        output_path = os.path.join(self.model_config.output_folder, task_folder)
        
        timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        session_folder = os.path.join(output_path, f"session_{timestamp}")
        
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(session_folder, exist_ok=True)
        
        return output_path, session_folder
    
    def _create_training_environments(self, output_path: str) -> Tuple[VecNormalize, VecNormalize]:
        """Create training and evaluation environments."""
        env_kwargs = EnvironmentFactory.get_env_kwargs(
            self.training_config.task, 
            self.env_config, 
            self.training_config
        )
        
        # Create vectorized environments
        train_env_raw = make_vec_env(
            lambda: EnvironmentFactory.create_environment(
                self.training_config.task, 
                self.training_config.trajectory_type, 
                **env_kwargs
            ),
            n_envs=1
        )
        
        eval_env_raw = make_vec_env(
            lambda: EnvironmentFactory.create_environment(
                self.training_config.task, 
                self.training_config.trajectory_type, 
                **env_kwargs
            ),
            n_envs=1
        )
        
        # Setup VecNormalize
        existing_stats_path = self.model_manager.get_stats_path(
            self.training_config.task, 
            self.training_config.trajectory_type
        )
        
        train_env, eval_env = self._setup_vec_normalize(
            train_env_raw, eval_env_raw, existing_stats_path
        )
        
        print(f'[INFO] Action space: {train_env.action_space}')
        print(f'[INFO] Observation space: {train_env.observation_space}')
        
        return train_env, eval_env
    
    def _setup_vec_normalize(
        self, 
        train_env_raw, 
        eval_env_raw, 
        existing_stats_path: str
    ) -> Tuple[VecNormalize, VecNormalize]:
        """Setup VecNormalize with existing or new statistics."""
        if (self.model_config.load_model and 
            os.path.exists(self.model_config.load_model) and 
            os.path.exists(existing_stats_path)):
            
            print(f"📊 Loading existing VecNormalize stats from: {existing_stats_path}")
            try:
                train_env = VecNormalize.load(existing_stats_path, train_env_raw)
                train_env.training = True
                train_env.norm_reward = True
                
                eval_env = VecNormalize.load(existing_stats_path, eval_env_raw)
                eval_env.training = False
                eval_env.norm_reward = False
                
                print("✅ Existing VecNormalize stats loaded successfully!")
                print(f"   Observation count: {train_env.obs_rms.count}")
                print(f"   Return count: {train_env.ret_rms.count}")
                
            except Exception as e:
                print(f"⚠️  Could not load existing VecNormalize stats: {e}")
                print("   Creating new VecNormalize...")
                train_env = VecNormalize(train_env_raw, norm_obs=True, norm_reward=True, gamma=0.99)
                eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False, training=False, gamma=0.99)
        else:
            print("📊 Creating new VecNormalize statistics...")
            train_env = VecNormalize(train_env_raw, norm_obs=True, norm_reward=True, gamma=0.99)
            eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False, training=False, gamma=0.99)
        
        return train_env, eval_env
    
    def _setup_model(self, train_env, output_path: str) -> PPO:
        """Setup PPO model (create new or load existing)."""
        if (self.model_config.load_model and 
            os.path.exists(self.model_config.load_model)):
            
            return self.model_manager.load_model_with_compatibility_check(
                self.model_config.load_model,
                train_env,
                self.training_config.learning_rate,
                self.training_config.task
            )
        else:
            print("🎯 Creating new PPO model...")
            model = self.model_manager.create_ppo_model(
                train_env, 
                self.training_config.learning_rate, 
                self.training_config.task
            )
            
            print(f"   New model info:")
            print(f"   Learning rate: {model.lr_schedule(1.0)}")
            print(f"   Observation space: {model.observation_space}")
            print(f"   Action space: {model.action_space}")
            
            return model
    
    def _setup_callbacks(self, eval_env, session_folder: str) -> EvalCallback:
        """Setup training callbacks."""
        target_reward = EnvironmentFactory.get_target_reward(
            self.training_config.task, 
            self.env_config.act_type
        )
        
        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=target_reward,
            verbose=1
        )
        
        return EvalCallback(
            eval_env,
            callback_on_new_best=callback_on_best,
            verbose=1,
            best_model_save_path=session_folder + '/',
            log_path=session_folder + '/',
            eval_freq=self.training_config.eval_freq,
            deterministic=True,
            render=False,
        )
    
    def _train_model(self, model: PPO, eval_callback: EvalCallback):
        """Train the PPO model."""
        total_timesteps = self.training_config.episodes * 1000
        print(f"🎮 Training for {total_timesteps} timesteps...")
        
        if (self.model_config.load_model and 
            os.path.exists(self.model_config.load_model)):
            print("   Continuing training from loaded model...")
        else:
            print("   Starting training from scratch...")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            log_interval=100,
            reset_num_timesteps=False if self.model_config.load_model else True
        )
    
    def _post_training_cleanup(
        self, 
        train_env, 
        eval_env, 
        output_path: str, 
        session_folder: str, 
        training_start_time: float
    ):
        """Handle post-training cleanup and notifications."""
        # Print final environment statistics if available
        if hasattr(train_env.envs[0], 'print_final_stats'):
            print("\n" + "="*50)
            print("📈 TRAINING COMPLETED - FINAL STATISTICS")
            train_env.envs[0].print_final_stats()
        
        # Save timing statistics if available
        if hasattr(train_env.envs[0], 'get_timing_stats'):
            stats = train_env.envs[0].get_timing_stats()
            stats_file = os.path.join(output_path, 'timing_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"📊 Timing statistics saved to: {stats_file}")
        
        # Save models and statistics
        self._save_models_and_stats(output_path, session_folder, train_env)
        
        # Close environments
        train_env.close()
        eval_env.close()
        
        # Save training info and plot progress
        self._save_training_info_and_plot(output_path, session_folder, training_start_time)
        
        print(f"\n✅ Training completed!")
        print(f"   Best model saved to: {os.path.join(output_path, 'best_model.zip')}")
        print(f"   Session details: {session_folder}")
    
    def _save_models_and_stats(self, output_path: str, session_folder: str, train_env):
        """Save best model and VecNormalize statistics."""
        session_best_path = os.path.join(session_folder, 'best_model.zip')
        main_best_path = os.path.join(output_path, 'best_model.zip')
        
        if os.path.exists(session_best_path):
            import shutil
            shutil.copy2(session_best_path, main_best_path)
            print(f"   Best model saved to: {main_best_path}")
        else:
            print("    No best model found in session folder")
        
        # Save VecNormalize statistics
        main_stats_path = os.path.join(output_path, "vec_normalize.pkl")
        session_stats_path = os.path.join(session_folder, "vec_normalize.pkl")
        
        train_env.save(main_stats_path)
        print(f"✅ Main stats saved to: {main_stats_path}")
        train_env.save(session_stats_path)
        print(f"✅ Session stats saved to: {session_stats_path}")
    
    def _save_training_info_and_plot(self, output_path: str, session_folder: str, training_start_time: float):
        """Save training information and plot progress."""
        timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        target_reward = EnvironmentFactory.get_target_reward(
            self.training_config.task, 
            self.env_config.act_type
        )
        
        training_info = {
            'task': self.training_config.task,
            'trajectory_type': (self.training_config.trajectory_type 
                              if self.training_config.task == "trajectory" else None),
            'episodes': self.training_config.episodes,
            'learning_rate': self.training_config.learning_rate,
            'target_reward': target_reward,
            'timestamp': timestamp,
            'session_folder': session_folder,
            'total_timesteps': self.training_config.episodes * 1000,
            'enable_obstacles': (self.training_config.enable_obstacles 
                               if self.training_config.task == "unified" else None)
        }
        
        with open(os.path.join(output_path, 'training_info.json'), 'w') as f:
            json.dump(training_info, f, indent=2)
        
        # Plot training progress
        self._plot_training_progress(session_folder, output_path, target_reward, training_start_time)
    
    def _plot_training_progress(self, session_folder: str, output_path: str, target_reward: float, training_start_time: float):
        """Plot and save training progress."""
        eval_file = os.path.join(session_folder, 'evaluations.npz')
        if os.path.exists(eval_file):
            print("\n   Training progression:")
            with np.load(eval_file) as data:
                timesteps = data['timesteps']
                results = data['results']
                
                # Create plot
                plt.figure(figsize=(10, 6))
                plt.plot(timesteps, results.mean(axis=1), 'b-', label='Mean Reward')
                plt.fill_between(timesteps, 
                               results.mean(axis=1) - results.std(axis=1),
                               results.mean(axis=1) + results.std(axis=1),
                               alpha=0.2, color='blue')
                plt.axhline(y=target_reward, color='r', linestyle='--', 
                           label=f'Target ({target_reward})')
                plt.xlabel('Timesteps')
                plt.ylabel('Mean Reward')
                plt.title(f'Training Progress - {self.training_config.task.capitalize()} Task')
                plt.legend()
                plt.grid(True)
                
                # Save plot
                plt.savefig(os.path.join(session_folder, 'training_progress.png'), 
                           dpi=150, bbox_inches='tight')
                plt.savefig(os.path.join(output_path, 'latest_training_progress.png'), 
                           dpi=150, bbox_inches='tight')
                
                if not self.model_config.colab:
                    plt.show()
                
                # Send notification
                training_duration = time.time() - training_start_time
                final_performance = f"{results[-1][0]:.1f}" if len(results) > 0 else "未知"
                
                self.notification_manager.send_training_completion_notification(
                    task=self.training_config.task,
                    episodes=self.training_config.episodes,
                    final_reward=final_performance,
                    target_reward=target_reward,
                    training_time=training_duration,
                    enable_obstacles=(self.training_config.enable_obstacles 
                                    if self.training_config.task == "unified" else None)
                )
                
                # Print final statistics
                if len(results) > 0:
                    print(f"   Final performance: {results[-1][0]:.2f} reward")
                    print(f"   Target reward: {target_reward}")
                    print(f"{'✅ Target reached!' if results[-1][0] >= target_reward else '❌ Target not reached'}")
        else:
            # Send notification even without evaluation file
            training_duration = time.time() - training_start_time
            self.notification_manager.send_training_completion_notification(
                task=self.training_config.task,
                episodes=self.training_config.episodes,
                final_reward="未知",
                target_reward=target_reward,
                training_time=training_duration,
                enable_obstacles=(self.training_config.enable_obstacles 
                                if self.training_config.task == "unified" else None)
            )
    
    def _setup_evaluation_environment(self, model_path: str) -> Tuple[PPO, Any, Any]:
        """Setup environment and model for evaluation."""
        # Find normalization stats
        stats_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"VecNormalize stats not found at {stats_path}")
        
        # Load model
        model = PPO.load(model_path)
        print("✅ Model loaded successfully!")
        
        # Create evaluation environment kwargs
        env_kwargs = EnvironmentFactory.get_env_kwargs(
            self.training_config.task, 
            self.env_config, 
            self.training_config
        )
        env_kwargs.update({
            'gui': self.env_config.gui,
            'record': self.env_config.record_video,
            'randomize_init': True
        })
        
        print(f"🔧 Environment parameters: {env_kwargs}")
        
        # Create test environments
        test_env_raw = EnvironmentFactory.create_environment(
            self.training_config.task, 
            self.training_config.trajectory_type, 
            **env_kwargs
        )
        test_env = VecNormalize.load(stats_path, test_env_raw)
        test_env.training = False
        test_env.norm_reward = False
        
        # Create no-GUI environment for policy evaluation
        env_kwargs_nogui = env_kwargs.copy()
        env_kwargs_nogui.update({'gui': False, 'record': False})
        test_env_nogui_raw = EnvironmentFactory.create_environment(
            self.training_config.task, 
            self.training_config.trajectory_type, 
            **env_kwargs_nogui
        )
        test_env_nogui = VecNormalize.load(stats_path, test_env_nogui_raw)
        test_env_nogui.training = False
        test_env_nogui.norm_reward = False
        
        print(f"✅ Normalization stats loaded from {stats_path}")
        
        return model, test_env, test_env_nogui
    
    def _perform_evaluation(self, model: PPO, test_env, test_env_nogui):
        """Perform model evaluation with visualization."""
        # Evaluate policy performance
        print("   Evaluating policy performance...")
        mean_reward, std_reward = evaluate_policy(
            model, test_env_nogui, n_eval_episodes=10
        )
        print(f"   Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Create logger
        logger = Logger(
            logging_freq_hz=int(test_env.CTRL_FREQ),
            num_drones=1,
            output_folder=self.model_config.output_folder,
            colab=self.model_config.colab
        )
        
        # Run visualization
        self._run_evaluation_loop(model, test_env, logger)
        
        # Close environments
        test_env.close()
        test_env_nogui.close()
        
        # Save and plot results
        print(f"Saving results to: {self.model_config.output_folder}")
        logger.save()
        logger.save_as_csv(f"{self.training_config.task}_evaluation")
        
        if self.env_config.obs_type == ObservationType.KIN:
            logger.plot()
        
        print(f"\nMean evaluation reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    def _run_evaluation_loop(self, model: PPO, test_env, logger: Logger):
        """Run the main evaluation loop with visualization."""
        total_reward = 0
        step_count = 0
        episode_count = 0
        
        print(f"Running visualization for {self.env_config.duration_sec} seconds...")
        obs, info = test_env.reset(seed=int(time.time()), options={})
        episode_count += 1
        
        print(f"Starting evaluation episode {episode_count}...")
        if hasattr(test_env, 'TARGET_POS'):
            print(f"        Target position: [[{test_env.TARGET_POS[0]:.3f}, {test_env.TARGET_POS[1]:.3f}, {test_env.TARGET_POS[2]:.3f}]")
        
        if hasattr(test_env, '_getDroneStateVector'):
            initial_state = test_env._getDroneStateVector(0)
            initial_pos = initial_state[0:3]
            print(f"        Initial position: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}]")
        
        start = time.time()
        
        for i in range(int(self.env_config.duration_sec * test_env.CTRL_FREQ)):
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward
            step_count += 1
            
            # Log data
            self._log_step_data(logger, obs, action, i, test_env)
            
            # Print progress
            if i % (test_env.CTRL_FREQ * 2) == 0:
                self._print_evaluation_progress(i, test_env, reward, total_reward, step_count, info)
            
            # Render
            test_env.render()
            sync(i, start, test_env.CTRL_TIMESTEP)
            
            # Handle episode termination
            if terminated or truncated:
                episode_count += 1
                print(f"Episode {episode_count} ended!")
                print(f"   Terminated: {terminated}, Truncated: {truncated}")
                
                self._handle_episode_termination(test_env, terminated)
                obs, info = test_env.reset(seed=int(time.time()), options={})
        
        # Print final statistics
        print(f"\nFinal Results:")
        print(f"   Total episodes: {episode_count}")
        print(f"   Total steps: {step_count}")
        print(f"   Average reward per step: {total_reward/max(step_count, 1):.3f}")
    
    def _log_step_data(self, logger: Logger, obs, action, step: int, test_env):
        """Log step data for analysis."""
        if self.env_config.obs_type == ObservationType.KIN:
            obs_flat = obs.squeeze() if hasattr(obs, 'squeeze') else obs
            act_flat = action.squeeze() if hasattr(action, 'squeeze') else action
            
            if len(obs_flat) >= 12:
                state = np.hstack([
                    obs_flat[0:3],      # position
                    np.zeros(4),        # quaternion (placeholder)
                    obs_flat[3:15] if len(obs_flat) >= 15 else np.pad(obs_flat[3:], (0, max(0, 12-len(obs_flat[3:])))),
                    act_flat if hasattr(act_flat, '__len__') else [act_flat]
                ])
            else:
                state = np.pad(obs_flat, (0, max(0, 16-len(obs_flat))))
            
            logger.log(
                drone=0,
                timestamp=step / test_env.CTRL_FREQ,
                state=state,
                control=np.zeros(12)
            )
    
    def _print_evaluation_progress(self, step: int, test_env, reward: float, total_reward: float, step_count: int, info):
        """Print evaluation progress information."""
        avg_reward = total_reward / max(step_count, 1)
        print(f"    Time: {step/test_env.CTRL_FREQ:.1f}s, "
              f"Reward: {reward:.3f}, "
              f"Avg Reward: {avg_reward:.3f}")
        
        # Task-specific info
        if hasattr(info, 'get'):
            if self.training_config.task == "unified":
                distance = info.get('distance_to_target', 0)
                hover_time = info.get('time_at_target', 0)
                print(f"      Distance to target: {distance:.3f}m, Hover time: {hover_time:.1f}s")
            elif self.training_config.task == "trajectory":
                progress = info.get('trajectory_progress', 0)
                print(f"      Progress: {progress:.1%}")
            elif self.training_config.task == "obstacle":
                distance = info.get('distance_to_goal', 0)
                print(f"      Distance to goal: {distance:.2f}m")
    
    def _handle_episode_termination(self, test_env, terminated: bool):
        """Handle episode termination and print results."""
        if self.training_config.task == "unified":
            hover_time = getattr(test_env, 'time_at_target', 0)
            required_time = getattr(test_env, 'required_hover_time', 3.0)
            if terminated:
                print(f"   ✅ Task completed! Hovered for {hover_time:.1f}s (required: {required_time:.1f}s)")
            else:
                print(f"   ❌ Task not completed. Hover time: {hover_time:.1f}s (required: {required_time:.1f}s)")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description='Deep Reinforcement Learning for Drone Control (Refactored)')
    
    # Task parameters
    parser.add_argument('--task', default='hover', type=str,
                       choices=['hover', 'trajectory', 'obstacle', 'unified'],
                       help='Task to train/evaluate (default: hover)')
    parser.add_argument('--trajectory_type', default='circle', type=str,
                       choices=['circle', 'figure8', 'waypoints'],
                       help='Trajectory type for trajectory task (default: circle)')
    
    # Training parameters
    parser.add_argument('--train_mode', default=True, type=str2bool,
                       help='Whether to train (True) or evaluate (False) (default: True)')
    parser.add_argument('--episodes', default=1000, type=int,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--learning_rate', default=3e-4, type=float,
                       help='Learning rate for training (default: 3e-4)')
    parser.add_argument('--eval_freq', default=2000, type=int,
                       help='Evaluation frequency during training (default: 2000)')
    parser.add_argument('--enable_obstacles', default=True, type=str2bool,
                       help='Whether to enable obstacles for unified task (default: True)')
    
    # Environment parameters
    parser.add_argument('--obs_type', default=ObservationType('kin'), type=ObservationType,
                       help='Observation type (default: kin)')
    parser.add_argument('--act_type', default=ActionType('rpm'), type=ActionType,
                       help='Action type (default: rpm)')
    parser.add_argument('--gui', default=True, type=str2bool,
                       help='Whether to use PyBullet GUI (default: True)')
    parser.add_argument('--record_video', default=False, type=str2bool,
                       help='Whether to record video (default: False)')
    parser.add_argument('--duration_sec', default=30, type=int,
                       help='Duration for evaluation in seconds (default: 30)')
    
    # Model parameters
    parser.add_argument('--load_model', default=None, type=str,
                       help='Path to model to load (for continuing training or evaluation)')
    parser.add_argument('--output_folder', default='results', type=str,
                       help='Output folder for logs and models (default: results)')
    parser.add_argument('--colab', default=False, type=str2bool,
                       help='Whether running in Colab (default: False)')
    
    # Utility parameters
    parser.add_argument('--list_models', action='store_true',
                       help='List all available trained models')
    
    return parser


def main():
    """Main entry point."""
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
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Create trainer and run
    trainer = DroneRLTrainer(training_config, env_config, model_config)
    
    print(f"🎯 Task: {args.task}")
    if args.task == "trajectory":
        print(f"🛤️  Trajectory type: {args.trajectory_type}")
    print(f"🎮 Mode: {'Training' if args.train_mode else 'Evaluation'}")
    print(f"📁 Output folder: {args.output_folder}")
    
    if args.train_mode:
        result_folder = trainer.run_training()
        if not args.colab:
            print(f"\nTraining completed!")
            print(f"To list available models: python rl_refactored.py --list_models")
            print(f"To evaluate: python rl_refactored.py --task {args.task} --train_mode False --gui True")
    else:
        trainer.run_evaluation(args.load_model)


if __name__ == '__main__':
    main()
