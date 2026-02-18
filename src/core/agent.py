"""
DQN Agent
High-level interface for DQN-based power control agent
"""

import numpy as np
import torch
from dqn.trainer import DQNTrainer


class DQNAgent:
    """
    DQN Agent for power control

    Provides simple interface for:
    - Action selection
    - Learning from experience
    - Saving/loading models
    """

    def __init__(self,
                 user_id: int,
                 state_dim: int,
                 action_dim: int,
                 config_path: str = "config/dqn.yaml",
                 device: str = "cuda"):
        self.user_id = user_id
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Create DQN trainer
        self.trainer = DQNTrainer(
            state_dim=state_dim,
            action_dim=action_dim,
            config_path=config_path,
            device=device
        )

        # Episode tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.current_episode_reward = 0.0
        self.current_episode_step = 0
        self.total_steps = 0
        self.train_every = 16  # Train every N steps (avoids overhead of 250 train calls/ep)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action based on current state."""
        if training:
            action = self.trainer.select_action(state)
        else:
            action = self.trainer.policy_net.get_action(state, epsilon=0.0)
        return action
    
    def learn(self, state, action, reward, next_state, done) -> float:
        """Learn from transition. Returns training loss."""
        # Store transition
        self.trainer.store_transition(state, action, reward, next_state, done)

        # Update episode stats
        self.current_episode_reward += reward
        self.current_episode_step += 1
        self.total_steps += 1

        # Train every N steps (reduces CPU↔GPU transfer overhead)
        if self.total_steps % self.train_every == 0:
            loss = self.trainer.train_step()
        else:
            loss = 0.0

        # End of episode
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_steps.append(self.current_episode_step)
            self.trainer.episode_end()
            self.current_episode_reward = 0.0
            self.current_episode_step = 0

        return loss
    
    def get_epsilon(self) -> float:
        """Get current exploration rate"""
        return self.trainer.epsilon
    
    def set_epsilon(self, epsilon: float):
        """Set exploration rate"""
        self.trainer.set_epsilon(epsilon)
    
    def save(self, filepath: str):
        """Save agent model"""
        self.trainer.save_checkpoint(filepath)
    
    def load(self, filepath: str):
        """Load agent model"""
        self.trainer.load_checkpoint(filepath)
    
    def get_stats(self) -> dict:
        """Get agent statistics"""
        stats = self.trainer.get_training_stats()
        stats.update({
            'user_id': self.user_id,
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0.0,
            'avg_episode_steps': np.mean(self.episode_steps[-100:]) if len(self.episode_steps) > 0 else 0.0,
            'total_episodes': len(self.episode_rewards)
        })
        return stats

    def close(self):
        """Close agent resources."""
        pass
