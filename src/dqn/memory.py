"""
Experience Replay Memory
Stores and samples transitions for DQN training
"""

import numpy as np
from collections import deque, namedtuple
import random


# Transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    """
    Simple circular buffer for experience replay
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition"""
        self.memory.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch of transitions
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """Return current size of buffer"""
        return len(self.memory)
    
    def clear(self):
        """Clear all stored transitions"""
        self.memory.clear()


class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay
    
    Samples transitions based on their TD error
    Higher priority = sampled more frequently
    
    Reference: Schaul et al. "Prioritized Experience Replay" (2016)
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling correction (increases to 1.0 during training)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001  # Increase beta over time
        
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Store transition with maximum priority
        (New experiences have high priority initially)
        """
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        transition = Transition(state, action, reward, next_state, done)
        
        if self.size < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """
        Sample batch with prioritized sampling
        
        Returns:
            batch: List of transitions
            indices: Indices of sampled transitions (for priority update)
            weights: Importance sampling weights
        """
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # Sample transitions
        batch = [self.memory[idx] for idx in indices]
        
        # Compute importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Increase beta (anneal to 1.0)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors
        
        Args:
            indices: Indices of transitions to update
            td_errors: TD errors (|δ|) for each transition
        """
        for idx, td_error in zip(indices, td_errors):
            # Priority = |TD error| + small constant (avoid zero)
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self):
        return self.size
    
    def clear(self):
        self.memory.clear()
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0


def create_replay_memory(memory_type: str = "standard", 
                         capacity: int = 50000,
                         alpha: float = 0.6,
                         beta: float = 0.4):
    """
    Factory function to create replay memory
    
    Args:
        memory_type: "standard" or "prioritized"
        capacity: Buffer capacity
        alpha: Prioritization exponent (for PER)
        beta: Importance sampling correction (for PER)
        
    Returns:
        Replay memory instance
    """
    if memory_type == "standard":
        return ReplayMemory(capacity)
    elif memory_type == "prioritized":
        return PrioritizedReplayMemory(capacity, alpha, beta)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")
