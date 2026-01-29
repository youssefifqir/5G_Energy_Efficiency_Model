"""
DQN Trainer
Implements training loop with target network and Huber loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
import yaml

from .model import create_dqn_model
from .memory import create_replay_memory, Transition


class DQNTrainer:
    """
    DQN Training Manager
    
    Implements:
    - Double DQN (policy + target networks)
    - Huber loss for stability
    - Experience replay
    - ε-greedy exploration with decay
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 config_path: str = "config/dqn.yaml",
                 device: str = "cuda"):
        """
        Initialize DQN trainer
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            config_path: Path to DQN configuration
            device: "cuda" or "cpu"
        """
        # Load configuration
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)['dqn']
        
        self.config = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        # Create policy and target networks
        hidden_layers = cfg['network']['hidden_layers']
        self.policy_net = create_dqn_model(
            state_dim, 
            action_dim, 
            model_type="standard",
            hidden_layers=hidden_layers
        ).to(self.device)
        
        self.target_net = create_dqn_model(
            state_dim,
            action_dim,
            model_type="standard", 
            hidden_layers=hidden_layers
        ).to(self.device)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in eval mode
        
        # Optimizer
        lr = cfg['training']['learning_rate']
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay memory
        buffer_size = cfg['training']['buffer_size']
        self.memory = create_replay_memory("standard", capacity=buffer_size)
        
        # Training parameters
        self.batch_size = cfg['training']['batch_size']
        self.gamma = cfg['training']['gamma']
        self.target_update_freq = cfg['training']['target_update_freq']
        self.soft_update_tau = cfg['training']['soft_update_tau']
        self.min_buffer_size = cfg['training']['min_buffer_size']
        
        # Loss function (Huber loss for stability)
        self.loss_type = cfg['training']['loss_type']
        if self.loss_type == "huber":
            delta = cfg['training'].get('huber_delta', 1.0)
            self.criterion = nn.SmoothL1Loss()  # Huber loss in PyTorch
        else:
            self.criterion = nn.MSELoss()
        
        # Exploration parameters
        self.epsilon = cfg['training']['epsilon_start']
        self.epsilon_start = cfg['training']['epsilon_start']
        self.epsilon_end = cfg['training']['epsilon_end']
        self.epsilon_decay = cfg['training']['epsilon_decay']
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        self.losses = []
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using ε-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            Selected action index
        """
        return self.policy_net.get_action(state, epsilon=self.epsilon)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """
        Perform one training step
        
        Returns:
            Loss value
        """
        # Check if enough samples in buffer
        if len(self.memory) < self.min_buffer_size:
            return 0.0
        
        # Sample batch from memory
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute current Q-values: Q(s, a)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            # Double DQN: use policy net to select action, target net to evaluate
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            
            # Target: r + γ * Q_target(s', argmax_a Q_policy(s', a))
            # Set next_q to 0 if episode is done
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values.squeeze()
        
        # Compute loss (Huber or MSE)
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update target network (soft update)
        self._soft_update_target_network()
        
        # Update statistics
        self.training_step += 1
        self.losses.append(loss.item())
        
        return loss.item()
    
    def _soft_update_target_network(self):
        """
        Soft update target network parameters
        
        θ_target = τ * θ_policy + (1 - τ) * θ_target
        """
        tau = self.soft_update_tau
        
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                               self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)
    
    def hard_update_target_network(self):
        """
        Hard update: copy policy network weights to target network
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def set_epsilon(self, epsilon: float):
        """Manually set epsilon"""
        self.epsilon = max(self.epsilon_end, min(self.epsilon_start, epsilon))
    
    def episode_end(self):
        """Call at end of episode to decay epsilon"""
        self.episode_count += 1
        self.decay_epsilon()
    
    def save_checkpoint(self, filepath: str):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'losses': self.losses
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.losses = checkpoint['losses']
        
        print(f"Checkpoint loaded from {filepath}")
    
    def get_training_stats(self) -> dict:
        """Get training statistics"""
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'buffer_size': len(self.memory),
            'avg_loss': np.mean(self.losses[-100:]) if len(self.losses) > 0 else 0.0
        }
