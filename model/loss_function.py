# model/loss_function.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DQNLoss(nn.Module):
    """
    Loss function for DQN training.
    
    Computes the temporal difference (TD) loss between current Q-values
    and target Q-values based on the Bellman equation.
    """
    
    def __init__(self, loss_type: str = 'mse', huber_delta: float = 1.0):
        """
        Args:
            loss_type: Type of loss - 'mse' or 'huber'
            huber_delta: Delta parameter for Huber loss (only used if loss_type='huber')
        """
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss(beta=huber_delta)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'mse' or 'huber'")
    
    
    def forward(
        self,
        q_values: torch.Tensor,
        target_q_values: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the DQN loss.
        
        Args:
            q_values: Current Q-values from online network [batch_size]
            target_q_values: Target Q-values from target network [batch_size]
            weights: Optional importance sampling weights [batch_size] for prioritized replay
        
        Returns:
            Scalar loss tensor
        """
        # Compute element-wise loss
        if self.loss_type == 'mse':
            element_wise_loss = F.mse_loss(q_values, target_q_values, reduction='none')
        elif self.loss_type == 'huber':
            element_wise_loss = F.smooth_l1_loss(q_values, target_q_values, beta=self.huber_delta, reduction='none')
        
        # Apply importance sampling weights if provided (for prioritized experience replay)
        if weights is not None:
            element_wise_loss = element_wise_loss * weights
        
        # Return mean loss
        return element_wise_loss.mean()


class DoubleDQNLoss(nn.Module):
    """
    Loss function for Double DQN training.
    
    Uses the online network to select actions and the target network to evaluate them,
    reducing overestimation bias.
    """
    
    def __init__(self, loss_type: str = 'mse', huber_delta: float = 1.0):
        """
        Args:
            loss_type: Type of loss - 'mse' or 'huber'
            huber_delta: Delta parameter for Huber loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss(beta=huber_delta)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    
    
    def compute_target(
        self,
        online_network: nn.Module,
        target_network: nn.Module,
        next_states: torch.Tensor,
        candidate_actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float
    ) -> torch.Tensor:
        """
        Compute Double DQN target values.
        
        Args:
            online_network: Online Q-network (for action selection)
            target_network: Target Q-network (for value estimation)
            next_states: Next states [batch_size, state_dim]
            candidate_actions: All candidate action features [batch_size, num_candidates, action_dim]
            rewards: Rewards [batch_size]
            dones: Done flags [batch_size]
            gamma: Discount factor
        
        Returns:
            Target Q-values [batch_size]
        """
        with torch.no_grad():
            # Use online network to select best actions
            online_q_values = []
            for i in range(len(next_states)):
                q_vals = online_network.q_values_for_candidates(
                    next_states[i:i+1],
                    candidate_actions[i:i+1]
                )
                online_q_values.append(q_vals)
            online_q_values = torch.cat(online_q_values, dim=0)  # [batch_size, num_candidates]
            
            # Select best actions
            best_actions_idx = online_q_values.argmax(dim=1)  # [batch_size]
            
            # Use target network to evaluate selected actions
            target_q_values = []
            for i in range(len(next_states)):
                best_action = candidate_actions[i, best_actions_idx[i]]
                q_val = target_network(next_states[i:i+1], best_action.unsqueeze(0))
                target_q_values.append(q_val)
            target_q_values = torch.cat(target_q_values, dim=0)  # [batch_size]
            
            # Bellman target
            target = rewards + gamma * target_q_values * (~dones).float()
        
        return target
    
    
    def forward(
        self,
        q_values: torch.Tensor,
        target_q_values: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the Double DQN loss.
        
        Args:
            q_values: Current Q-values from online network [batch_size]
            target_q_values: Target Q-values (pre-computed) [batch_size]
            weights: Optional importance sampling weights [batch_size]
        
        Returns:
            Scalar loss tensor
        """
        # Compute element-wise loss
        if self.loss_type == 'mse':
            element_wise_loss = F.mse_loss(q_values, target_q_values, reduction='none')
        elif self.loss_type == 'huber':
            element_wise_loss = F.smooth_l1_loss(q_values, target_q_values, beta=self.huber_delta, reduction='none')
        
        # Apply importance sampling weights if provided
        if weights is not None:
            element_wise_loss = element_wise_loss * weights
        
        return element_wise_loss.mean()


class TDLoss(nn.Module):
    """
    Generic Temporal Difference (TD) loss for value-based RL.
    """
    
    def __init__(self, gamma: float = 0.99, loss_type: str = 'mse'):
        """
        Args:
            gamma: Discount factor
            loss_type: 'mse' or 'huber'
        """
        super().__init__()
        self.gamma = gamma
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
    
    
    def compute_td_target(
        self,
        rewards: torch.Tensor,
        next_q_values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute TD target: r + γ * max_a' Q(s', a') * (1 - done)
        
        Args:
            rewards: Rewards [batch_size]
            next_q_values: Q-values for next states [batch_size]
            dones: Terminal flags [batch_size]
        
        Returns:
            TD targets [batch_size]
        """
        return rewards + self.gamma * next_q_values * (~dones).float()
    
    
    def forward(
        self,
        current_q: torch.Tensor,
        rewards: torch.Tensor,
        next_q: torch.Tensor,
        dones: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute TD loss.
        
        Args:
            current_q: Current Q-values [batch_size]
            rewards: Rewards [batch_size]
            next_q: Next Q-values [batch_size]
            dones: Done flags [batch_size]
            weights: Optional importance weights [batch_size]
        
        Returns:
            Scalar loss
        """
        # Compute target
        target = self.compute_td_target(rewards, next_q, dones)
        
        # Compute loss
        loss = self.loss_fn(current_q, target.detach())
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()
