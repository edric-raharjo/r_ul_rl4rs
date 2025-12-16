# model/dqn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import os

class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return x

class CatalogQNetwork(nn.Module):
    """
    State -> Q-values over a fixed catalog of items (discrete actions).

    - num_items: total number of discrete item IDs (action space size)
    - forward(state): returns [B, num_items] Q(s, a) for all a
    """

    def __init__(self, state_dim: int, num_items: int, hidden_dim: int = 256):
        super().__init__()
        self.state_encoder = StateEncoder(state_dim, hidden_dim)
        self.fc_q = nn.Linear(hidden_dim, num_items)
        self.num_items = num_items

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: [B, state_dim]
        returns: q_all [B, num_items]
        """
        s_emb = self.state_encoder(state)      # [B, H]
        q_all = self.fc_q(s_emb)              # [B, num_items]
        return q_all

    @torch.no_grad()
    def q_values_for_candidates(
        self,
        state: torch.Tensor,                  # [B, state_dim]
        candidate_ids: torch.Tensor,          # [B, K] int64 indices into catalog
        candidate_mask: Optional[torch.Tensor] = None,  # [B, K] bool
    ) -> torch.Tensor:
        """
        Convenience: get Q(s, a) for a set of candidate item IDs.

        - state: [B, state_dim]
        - candidate_ids: [B, K] (each entry in [0, num_items-1])
        - candidate_mask: optional [B, K] bool; False -> invalid -> -inf
        returns: q_all [B, K]
        """
        B, K = candidate_ids.shape

        q_catalog = self.forward(state)       # [B, num_items]
        # gather Q for each candidate ID
        q_all = q_catalog.gather(1, candidate_ids)  # [B, K]

        if candidate_mask is not None:
            q_all = q_all.masked_fill(~candidate_mask, float("-inf"))

        return q_all

def create_q_networks(
    state_dim: int,
    num_items: int,
    hidden_dim: int = 256,
    device: str = "cuda",
) -> Tuple[CatalogQNetwork, CatalogQNetwork]:
    q = CatalogQNetwork(state_dim, num_items, hidden_dim).to(device)
    q_target = CatalogQNetwork(state_dim, num_items, hidden_dim).to(device)
    q_target.load_state_dict(q.state_dict())
    q_target.eval()
    return q, q_target


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float = 0.005):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def hard_update(target_net: nn.Module, source_net: nn.Module):
    target_net.load_state_dict(source_net.state_dict())

def save_model(model: nn.Module, save_dir: str, filename: str):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, filename))

def load_model(model: nn.Module, save_dir: str, filename: str, map_location: str = "cpu"):
    model.load_state_dict(torch.load(os.path.join(save_dir, filename), map_location=map_location))
