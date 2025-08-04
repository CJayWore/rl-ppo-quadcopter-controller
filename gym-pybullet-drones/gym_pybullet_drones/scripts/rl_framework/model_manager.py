"""
Model management for loading, saving, and configuring PPO models.
"""

import os
import json
from typing import Optional
from stable_baselines3 import PPO

from .environment import EnvironmentFactory


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
        
        # ========================= M1 Apple Silicon Configuration =========================
        # return PPO(
        #     'MlpPolicy',
        #     env,
        #     learning_rate=learning_rate,
        #     n_steps=2048,
        #     batch_size=128,
        #     n_epochs=10,
        #     gamma=0.99,
        #     gae_lambda=0.95,
        #     clip_range=0.2,
        #     policy_kwargs=policy_kwargs,
        #     verbose=1,
        #     device='auto',
        #     ent_coef=0.01,
        #     vf_coef=0.5,
        #     max_grad_norm=0.5
        # )
        
        # ========================= RTX 3090 Configuration =========================
        return PPO(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            n_steps=4096,               
            batch_size=1024,             
            n_epochs=15,                
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device='auto',
            ent_coef=0.008,
            vf_coef=0.5,
            max_grad_norm=0.5,
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
        
        # 只检查 unified 任务
        tasks = ['unified']
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
            print("💡 Train a model first with: python rl_modular.py --task hover --train_mode True")
