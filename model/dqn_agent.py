# model/dqn_agent.py (UPDATED)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for Recommendation System"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 128,
        loss_type: str = 'mse',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            state_dim: Dimension of state
            action_dim: Dimension of action features
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            tau: Soft update parameter for target network
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            loss_type: Type of loss function ('mse' or 'huber')
            device: 'cuda' or 'cpu'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        
        # Import QNetwork and Loss
        from model.dqn_model import QNetwork
        from model.loss_function import DQNLoss
        
        # Q-network and target network
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = DQNLoss(loss_type=loss_type)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.total_steps = 0
        self.losses = []
    
    
    def select_action(
        self,
        state: torch.Tensor,
        candidate_items: torch.Tensor,
        epsilon: float = 0.0
    ) -> Tuple[int, torch.Tensor]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: State tensor [state_dim]
            candidate_items: All candidate item features [num_items, action_dim]
            epsilon: Exploration rate
        
        Returns:
            (selected_item_idx, selected_item_features)
        """
        if random.random() < epsilon:
            # Random exploration
            item_idx = random.randint(0, len(candidate_items) - 1)
            return item_idx, candidate_items[item_idx]
        
        # Greedy exploitation
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            candidate_items = candidate_items.to(self.device)
            
            # Evaluate Q-values for all candidates
            q_values = self.q_network.q_values_for_candidates(
                state, 
                candidate_items.unsqueeze(0)
            )
            
            # Select item with highest Q-value
            item_idx = q_values.squeeze(0).argmax().item()
            
            return item_idx, candidate_items[item_idx].cpu()
    
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q-values
        current_q = self.q_network(states, actions)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states, actions)
            target_q = rewards + self.gamma * next_q * (~dones).float()
        
        # Compute loss using custom loss function
        loss = self.criterion(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update_target()
        
        # Track stats
        self.total_steps += 1
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    
    def soft_update_target(self):
        """Soft update target network parameters"""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    
    def hard_update_target(self):
        """Hard update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    
    def save(self, path: str):
        """Save agent checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'losses': self.losses
        }, path)
        print(f"Saved checkpoint to {path}")
    
    
    def load(self, path: str):
        """Load agent checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.losses = checkpoint['losses']
        print(f"Loaded checkpoint from {path}")
