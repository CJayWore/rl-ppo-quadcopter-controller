import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class ActorCritic(nn.Module):
    """Actor-Critic neural network for PPO algorithm."""
    
    def __init__(self, state_dim=12, action_dim=4, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Actor network (policy) - 输出 RPM 值
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim),
            nn.Sigmoid()  # 输出 [0, 1]，后续映射到 [0, MAX_RPM]
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 小的权重初始化
        self._init_weights()
        
        # Action standard deviation (learnable parameter)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -2.0)
        
    def _init_weights(self):
        """Initialize network weights to small values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, state):
        features = self.feature_extractor(state)
        action_mean = self.actor(features)
        value = self.critic(features)
        return action_mean, value
    
    def get_action_and_value(self, state, action=None):
        """Get action distribution and value."""
        mean, value = self.forward(state)
        std = torch.exp(self.log_std.clamp(-20, 2))
        dist = torch.distributions.Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        
        return action, log_prob, entropy, value.squeeze()

class ReplayBuffer:
    """Experience replay buffer for storing transitions."""
    
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, log_prob, value):
        self.buffer.append((state, action, reward, next_state, done, log_prob, value))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, log_prob, value = map(np.array, zip(*batch))
        return state, action, reward, next_state, done, log_prob, value
    
    def __len__(self):
        return len(self.buffer)

class DRLControl(BaseControl):
    """Deep Reinforcement Learning control class for Crazyflies using PPO."""

    def __init__(self,
                 drone_model: DroneModel,
                 g: float = 9.8,
                 learning_rate: float = 1e-4,
                 device: str = "cpu",
                 max_rpm: float = 21702.1):
        """Initialize the DRL controller with PPO algorithm."""
        super().__init__(drone_model=drone_model, g=g)
        
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] DRLControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.training = True
        self.MAX_RPM = max_rpm
        
        # State and action dimensions
        self.state_dim = 12  # [pos_error(3), vel_error(3), orientation(3), angular_vel(3)]
        self.action_dim = 4  # [rpm1, rpm2, rpm3, rpm4]
        
        # Neural network
        self.actor_critic = ActorCritic(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # PPO hyperparameters
        self.clip_epsilon = 0.1
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.update_epochs = 5
        self.batch_size = 32
        
        # Experience buffer
        self.buffer = ReplayBuffer(capacity=5000)
        self.episode_rewards = []
        self.episode_steps = 0
        
        # 基础 RPM (悬停时的 RPM)
        base_thrust = self.GRAVITY / (4 * self.KF)
        self.BASE_RPM = math.sqrt(base_thrust / self.KF)
        
        self.reset()

    def reset(self):
        """Reset the control class state."""
        super().reset()
        self.episode_steps = 0
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None
        self.episode_reward = 0

    def set_training_mode(self, training=True):
        """Set training mode."""
        self.training = training
        if training:
            self.actor_critic.train()
        else:
            self.actor_critic.eval()

    def _get_state(self, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_rpy):
        """Convert drone state to neural network input."""
        # Position and velocity errors
        pos_error = target_pos - cur_pos
        vel_error = np.zeros(3) - cur_vel  # Target velocity is zero (hover)
        
        # Current orientation (roll, pitch, yaw)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        rpy_error = target_rpy - cur_rpy
        
        # Normalize angular velocity
        normalized_ang_vel = cur_ang_vel / 10.0
        
        # Combine into state vector
        state = np.concatenate([
            pos_error,      # 3 elements
            vel_error,      # 3 elements  
            rpy_error,      # 3 elements
            normalized_ang_vel  # 3 elements
        ])
        
        # Normalize state for better learning
        state = np.clip(state, -5, 5)
        
        return state.astype(np.float32)

    def _compute_reward(self, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_rpy):
        """Compute reward signal for reinforcement learning."""
        # Position error penalty
        pos_error = np.linalg.norm(target_pos - cur_pos)
        pos_reward = -np.tanh(pos_error * 2)  # 使用tanh限制范围
        
        # Velocity penalty (encourage hovering)
        vel_magnitude = np.linalg.norm(cur_vel)
        vel_penalty = -0.1 * np.tanh(vel_magnitude)
        
        # Orientation error penalty
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        rpy_error = np.linalg.norm(target_rpy - cur_rpy)
        rpy_reward = -0.3 * np.tanh(rpy_error)
        
        # Angular velocity penalty (encourage stability)
        ang_vel_magnitude = np.linalg.norm(cur_ang_vel)
        ang_vel_penalty = -0.1 * np.tanh(ang_vel_magnitude)
        
        # Survival bonus
        survival_bonus = 0.1
        
        # Crash penalty
        crash_penalty = 0
        if cur_pos[2] < 0.05 or cur_pos[2] > 3.0:
            crash_penalty = -10
        if pos_error > 2.0:
            crash_penalty = -5
        
        # Height maintenance reward
        target_height = target_pos[2]
        height_error = abs(cur_pos[2] - target_height)
        height_reward = -0.3 * np.tanh(height_error)
        
        total_reward = pos_reward + vel_penalty + rpy_reward + ang_vel_penalty + survival_bonus + crash_penalty + height_reward
        total_reward = np.clip(total_reward, -20, 5)
        
        return total_reward

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)):
        """Compute DRL control action as RPMs."""
        self.control_counter += 1
        self.episode_steps += 1
        
        # Get current state
        state = self._get_state(cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_rpy)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action from neural network
        with torch.no_grad():
            action, log_prob, _, value = self.actor_critic.get_action_and_value(state_tensor)
            action = action.cpu().numpy().flatten()
            log_prob = log_prob.cpu().numpy()
            value = value.cpu().numpy()
        
        # Scale actions to RPM range
        # 从 [0, 1] 映射到 [BASE_RPM * 0.7, BASE_RPM * 1.3]
        min_rpm = self.BASE_RPM * 0.7
        max_rpm = self.BASE_RPM * 1.3
        rpm = min_rpm + action * (max_rpm - min_rpm)
        
        # Clip to valid range
        rpm = np.clip(rpm, 0, self.MAX_RPM)
        
        # Compute reward
        reward = self._compute_reward(cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_rpy)
        self.episode_reward += reward
        
        # Store experience in buffer (只在训练模式下)
        if self.training and self.last_state is not None:
            done = (cur_pos[2] < 0.05 or 
                   cur_pos[2] > 3.0 or 
                   np.linalg.norm(target_pos - cur_pos) > 2.0)
            
            self.buffer.push(
                self.last_state, self.last_action, reward, state, done,
                self.last_log_prob, self.last_value
            )
        
        # Update current step info
        self.last_state = state
        self.last_action = action
        self.last_log_prob = log_prob
        self.last_value = value
        
        # Training step
        if (self.training and len(self.buffer) > self.batch_size and 
            self.control_counter % 200 == 0):
            self._update_policy()
        
        # Calculate position error and yaw error for compatibility
        pos_e = target_pos - cur_pos
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        yaw_error = target_rpy[2] - cur_rpy[2]
        
        return rpm, pos_e, yaw_error

    def _update_policy(self):
        """Update the policy using PPO algorithm."""
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch from buffer
        states, actions, rewards, next_states, dones, old_log_probs, old_values = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        old_values = torch.FloatTensor(old_values).to(self.device)
        
        # Compute advantages using GAE
        with torch.no_grad():
            _, next_values = self.actor_critic(next_states)
            next_values = next_values.squeeze()
            
            advantages = torch.zeros_like(rewards)
            last_gae = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = ~dones[t]
                    next_value = next_values[t]
                else:
                    next_non_terminal = ~dones[t]
                    next_value = old_values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value * next_non_terminal - old_values[t]
                advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            
            returns = advantages + old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.update_epochs):
            # Get current policy outputs
            _, log_probs, entropy, values = self.actor_critic.get_action_and_value(states, actions)
            
            # Compute ratios
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Compute policy loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(values, returns)
            
            # Compute entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()

    def computeControlFromState(self,
                               control_timestep,
                               state,
                               target_pos,
                               target_rpy=np.zeros(3),
                               target_vel=np.zeros(3),
                               target_rpy_rates=np.zeros(3)):
        """Compute control action from drone state."""
        cur_pos = state[0:3]
        cur_quat = state[3:7]
        cur_vel = state[10:13]
        cur_ang_vel = state[13:16]
        
        return self.computeControl(
            control_timestep=control_timestep,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=target_pos,
            target_rpy=target_rpy,
            target_vel=target_vel,
            target_rpy_rates=target_rpy_rates
        )

    def save_model(self, path):
        """Save the trained model."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        print(f"Model loaded from {path}")

    def get_training_stats(self):
        """Get training statistics."""
        if len(self.episode_rewards) == 0:
            return {"episodes": 0, "avg_reward": 0, "max_reward": 0}
        
        return {
            "episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards[-100:]),
            "max_reward": np.max(self.episode_rewards),
            "total_steps": self.control_counter
        }