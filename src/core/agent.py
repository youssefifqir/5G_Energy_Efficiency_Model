"""
DQN Agent
High-level interface for DQN-based power control agent
"""

import numpy as np
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
        """
        Initialize agent
        
        Args:
            user_id: User identifier
            state_dim: State space dimension
            action_dim: Number of actions
            config_path: Configuration file path
            device: Computation device
        """
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
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action based on current state
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, use greedy
            
        Returns:
            Action index
        """
        if training:
            return self.trainer.select_action(state)
        else:
            # Greedy action (no exploration)
            return self.trainer.policy_net.get_action(state, epsilon=0.0)
    
    def learn(self, state, action, reward, next_state, done) -> float:
        """
        Learn from transition
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Episode termination flag
            
        Returns:
            Training loss (0.0 if not enough samples)
        """
        # Store transition
        self.trainer.store_transition(state, action, reward, next_state, done)
        
        # Update episode stats
        self.current_episode_reward += reward
        self.current_episode_step += 1
        
        # Train
        loss = self.trainer.train_step()
        
        # End of episode
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_steps.append(self.current_episode_step)
            self.trainer.episode_end()
            
            # Reset episode counters
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
