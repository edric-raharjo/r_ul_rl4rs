# model/dqn_model.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class StateEncoder(nn.Module):
    """
    Encoder network for state representation.
    Maps state vector to hidden representation.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int):
        """
        Args:
            state_dim: Dimension of input state
            hidden_dim: Dimension of hidden representation
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode state to hidden representation.
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            Hidden representation [batch_size, hidden_dim]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return x


class QNetwork(nn.Module):
    """
    Critic-only Deep Q-Network for recommendation.
    
    Takes state and item feature vectors as input and outputs Q-value.
    This is a state-action value network Q(s, a) where actions are
    represented as item feature vectors.
    """
    
    def __init__(self, state_dim: int, item_dim: int, hidden_dim: int = 256):
        """
        Args:
            state_dim: Dimension of state vector
            item_dim: Dimension of item (action) feature vector
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # State encoder
        self.state_encoder = StateEncoder(state_dim, hidden_dim)
        
        # Q-value head: takes concatenated [state_encoding, item_features]
        self.fc_q1 = nn.Linear(hidden_dim + item_dim, hidden_dim)
        self.fc_q2 = nn.Linear(hidden_dim, 1)
    
    
    def forward(self, state: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value for a (state, item) pair.
        
        Args:
            state: State tensor [batch_size, state_dim]
            item_vec: Item feature vector [batch_size, item_dim]
        
        Returns:
            Q-values [batch_size]
        """
        # Encode state
        s_emb = self.state_encoder(state)  # [batch_size, hidden_dim]
        
        # Concatenate state encoding with item features
        x = torch.cat([s_emb, item_vec], dim=-1)  # [batch_size, hidden_dim + item_dim]
        
        # Compute Q-value
        x = F.relu(self.fc_q1(x))  # [batch_size, hidden_dim]
        q = self.fc_q2(x).squeeze(-1)  # [batch_size]
        
        return q
    
    
    @torch.no_grad()
    def q_values_for_candidates(
        self,
        state: torch.Tensor,
        candidate_item_vecs: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Q-values for all candidate items given a state.
        Useful for action selection (choosing best item from candidates).
        
        Args:
            state: State tensor [batch_size, state_dim]
            candidate_item_vecs: Candidate item features [batch_size, num_candidates, item_dim]
            candidate_mask: Optional mask [batch_size, num_candidates] 
                           (True = valid, False = invalid)
        
        Returns:
            Q-values for all candidates [batch_size, num_candidates]
        """
        B, K, _ = candidate_item_vecs.shape
        
        # Expand state to match number of candidates
        state_rep = state.unsqueeze(1).expand(-1, K, -1)  # [B, K, state_dim]
        
        # Flatten batch and candidates for parallel Q-value computation
        state_flat = state_rep.reshape(B * K, -1)  # [B*K, state_dim]
        items_flat = candidate_item_vecs.reshape(B * K, -1)  # [B*K, item_dim]
        
        # Compute Q-values
        q_flat = self.forward(state_flat, items_flat)  # [B*K]
        
        # Reshape back to [batch_size, num_candidates]
        q_all = q_flat.reshape(B, K)  # [B, K]
        
        # Apply mask if provided (set invalid items to -inf)
        if candidate_mask is not None:
            q_all = q_all.masked_fill(~candidate_mask, float("-inf"))
        
        return q_all


def create_q_networks(
    state_dim: int,
    item_dim: int,
    hidden_dim: int = 256,
    device: str = "cuda"
) -> Tuple[QNetwork, QNetwork]:
    """
    Create Q-network and target Q-network.
    
    Args:
        state_dim: State dimension
        item_dim: Item feature dimension
        hidden_dim: Hidden layer dimension
        device: Device to place networks on
    
    Returns:
        (q_network, q_target_network)
    """
    q = QNetwork(state_dim, item_dim, hidden_dim).to(device)
    q_target = QNetwork(state_dim, item_dim, hidden_dim).to(device)
    q_target.load_state_dict(q.state_dict())
    q_target.eval()
    
    return q, q_target


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float = 0.005):
    """
    Soft update target network parameters.
    θ_target = τ * θ_source + (1 - τ) * θ_target
    
    Args:
        target_net: Target network to update
        source_net: Source network to copy from
        tau: Soft update coefficient
    """
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update(target_net: nn.Module, source_net: nn.Module):
    """
    Hard update target network (copy all parameters).
    
    Args:
        target_net: Target network to update
        source_net: Source network to copy from
    """
    target_net.load_state_dict(source_net.state_dict())


def save_model(model: nn.Module, save_dir: str, filename: str):
    """
    Save model state dict.
    
    Args:
        model: Model to save
        save_dir: Directory to save in
        filename: Filename for the checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, filename))


def load_model(model: nn.Module, save_dir: str, filename: str, map_location: str = "cpu"):
    """
    Load model state dict.
    
    Args:
        model: Model to load into
        save_dir: Directory containing checkpoint
        filename: Checkpoint filename
        map_location: Device to map tensors to
    """
    model.load_state_dict(
        torch.load(os.path.join(save_dir, filename), map_location=map_location)
    )


# Example usage
if __name__ == '__main__':
    # Test the Q-network
    batch_size = 32
    state_dim = 100  # max_history=50 -> 50*2
    item_dim = 20    # 1 + 19 genres
    num_candidates = 1000
    
    # Create network
    q_net = QNetwork(state_dim, item_dim, hidden_dim=256)
    
    # Test forward pass with single item
    states = torch.randn(batch_size, state_dim)
    items = torch.randn(batch_size, item_dim)
    q_values = q_net(states, items)
    print(f"Q-values shape: {q_values.shape}")  # [32]
    
    # Test forward pass with multiple candidates
    candidate_items = torch.randn(batch_size, num_candidates, item_dim)
    q_all = q_net.q_values_for_candidates(states, candidate_items)
    print(f"Q-values for all candidates shape: {q_all.shape}")  # [32, 1000]
    
    # Test with mask
    mask = torch.rand(batch_size, num_candidates) > 0.5
    q_masked = q_net.q_values_for_candidates(states, candidate_items, mask)
    print(f"Q-values with mask shape: {q_masked.shape}")  # [32, 1000]
    
    print("\nQ-Network test passed! ✓")
