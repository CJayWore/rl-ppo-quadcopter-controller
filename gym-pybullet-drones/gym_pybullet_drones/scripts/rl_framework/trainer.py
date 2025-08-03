"""
Main trainer class for drone reinforcement learning.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import time
import json
from datetime import datetime
from typing import Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize



from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.enums import ObservationType

from .config import TrainingConfig, EnvironmentConfig, ModelConfig
from .environment import EnvironmentFactory
from .model_manager import ModelManager
from .notifications import NotificationManager


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
            n_envs=16
        )
        
        eval_env_raw = make_vec_env(
            lambda: EnvironmentFactory.create_environment(
                self.training_config.task, 
                self.training_config.trajectory_type, 
                **env_kwargs
            ),
            n_envs=8
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
        test_env_raw = make_vec_env(
            lambda: EnvironmentFactory.create_environment(
                self.training_config.task, 
                self.training_config.trajectory_type, 
                **env_kwargs
            ),
            n_envs=1
        )
        test_env = VecNormalize.load(stats_path, test_env_raw)
        test_env.training = False
        test_env.norm_reward = False
        
        # Create no-GUI environment for policy evaluation
        env_kwargs_nogui = env_kwargs.copy()
        env_kwargs_nogui.update({'gui': False, 'record': False})
        test_env_nogui_raw = make_vec_env(
            lambda: EnvironmentFactory.create_environment(
            self.training_config.task, 
            self.training_config.trajectory_type, 
            **env_kwargs_nogui
            ),
            n_envs=1
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
            logging_freq_hz=int(test_env.envs[0].CTRL_FREQ),
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
        obs = test_env.reset()
        episode_count += 1
        
        print(f"Starting evaluation episode {episode_count}...")
        if hasattr(test_env.envs[0], 'TARGET_POS'):
            target_pos = test_env.envs[0].TARGET_POS
            print(f"        Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        if hasattr(test_env.envs[0], '_getDroneStateVector'):
            initial_state = test_env.envs[0]._getDroneStateVector(0)
            initial_pos = initial_state[0:3]
            print(f"        Initial position: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}]")
        
        start = time.time()
        
        ctrl_freq = test_env.envs[0].CTRL_FREQ
        for i in range(int(self.env_config.duration_sec * ctrl_freq)):
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = test_env.step(action)
            
            reward_scalar = reward[0] if hasattr(reward, '__len__') and len(reward) > 0 else reward
            done_scalar = done[0] if hasattr(done, '__len__') and len(done) > 0 else done
            
            total_reward += reward_scalar
            step_count += 1
            
            # Log data
            self._log_step_data(logger, obs, action, i, test_env)
            
            # Print progress
            if i % (ctrl_freq * 2) == 0:
                self._print_evaluation_progress(i, test_env, reward, total_reward, step_count, info)
            
            # Render
            test_env.render()
            sync(i, start, test_env.envs[0].CTRL_TIMESTEP)
            
            # Handle episode termination
            if done_scalar:
                episode_count += 1
                print(f"Episode {episode_count} ended!")
                print(f"   Done: {done_scalar}")
                
                self._handle_episode_termination(test_env, done_scalar)
                obs = test_env.reset()
        
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
                timestamp=step / test_env.envs[0].CTRL_FREQ,
                state=state,
                control=np.zeros(12)
            )
    
    def _print_evaluation_progress(self, step: int, test_env, reward, total_reward, step_count: int, info):
        """Print evaluation progress information."""
        avg_reward = total_reward / max(step_count, 1)
        ctrl_freq = test_env.envs[0].CTRL_FREQ
        
        reward_scalar = reward[0] if hasattr(reward, '__len__') and len(reward) > 0 else reward
        avg_reward_scalar = avg_reward[0] if hasattr(avg_reward, '__len__') and len(avg_reward) > 0 else avg_reward
        
        print(f"    Time: {step/ctrl_freq:.1f}s, "
            f"Reward: {reward_scalar:.3f}, "
            f"Avg Reward: {avg_reward_scalar:.3f}")
        
        # Task-specific info
        info_dict = info[0] if isinstance(info, (list, np.ndarray)) and len(info) > 0 else info
        
        if hasattr(info_dict, 'get') or isinstance(info_dict, dict):
            if self.training_config.task == "unified":
                distance = info_dict.get('distance_to_target', 0)
                hover_time = info_dict.get('time_at_target', 0)
                print(f"      Distance to target: {distance:.3f}m, Hover time: {hover_time:.1f}s")

    
    def _handle_episode_termination(self, test_env, done: bool):
        """Handle episode termination and print results."""
        if self.training_config.task == "unified":
            # 修复7: 使用 test_env.envs[0] 访问底层环境属性
            hover_time = getattr(test_env.envs[0], 'time_at_target', 0)
            required_time = getattr(test_env.envs[0], 'required_hover_time', 3.0)
            if done:
                print(f"   ✅ Task completed! Hovered for {hover_time:.1f}s (required: {required_time:.1f}s)")
            else:
                print(f"   ❌ Task not completed. Hover time: {hover_time:.1f}s (required: {required_time:.1f}s)")