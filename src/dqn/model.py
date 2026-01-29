"""
Deep Q-Network Model
Neural network architecture for Q-value approximation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for power control
    
    Architecture: Input → FC1 → ReLU → FC2 → ReLU → FC3 → ReLU → Output
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list = [128, 256, 128]):
        """
        Initialize DQN network
        
        Args:
            state_dim: Dimension of state space (5 in our case)
            action_dim: Number of discrete actions (power levels)
            hidden_layers: List of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer (Q-values for each action)
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            Q-values for each action [batch_size, action_dim]
        """
        return self.network(state)
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state (numpy array)
            epsilon: Exploration probability
            
        Returns:
            Selected action index
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.network[0].weight.device)
            q_values = self.forward(state_tensor)
            action = q_values.argmax(dim=1).item()
        return action


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture (optional enhancement)
    
    Separates Q-value into:
    - V(s): State value
    - A(s,a): Advantage of each action
    
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list = [128, 256]):
        super(DuelingDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single value output
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # Advantage for each action
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Dueling DQN
        
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        """
        # Extract features
        features = self.feature_layers(state)
        
        # Compute value and advantages
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine using dueling architecture formula
        # Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy"""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.feature_layers[0].weight.device)
            q_values = self.forward(state_tensor)
            action = q_values.argmax(dim=1).item()
        return action


def create_dqn_model(state_dim: int, 
                     action_dim: int, 
                     model_type: str = "standard",
                     hidden_layers: list = [128, 256, 128]) -> nn.Module:
    """
    Factory function to create DQN models
    
    Args:
        state_dim: State space dimension
        action_dim: Action space dimension
        model_type: "standard" or "dueling"
        hidden_layers: Hidden layer sizes
        
    Returns:
        DQN model
    """
    if model_type == "standard":
        return DQNNetwork(state_dim, action_dim, hidden_layers)
    elif model_type == "dueling":
        return DuelingDQN(state_dim, action_dim, hidden_layers[:2])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
