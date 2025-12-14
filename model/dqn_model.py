# model/dqn_model.py
import os
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, item_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_encoder = StateEncoder(state_dim, hidden_dim)
        self.fc_q1 = nn.Linear(hidden_dim + item_dim, hidden_dim)
        self.fc_q2 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor:
        s_emb = self.state_encoder(state)
        x = torch.cat([s_emb, item_vec], dim=-1)
        x = F.relu(self.fc_q1(x))
        q = self.fc_q2(x).squeeze(-1)
        return q

    @torch.no_grad()
    def q_values_for_candidates(self,
                               state: torch.Tensor,
                               candidate_item_vecs: torch.Tensor,
                               candidate_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, K, _ = candidate_item_vecs.shape
        state_rep = state.unsqueeze(1).expand(-1, K, -1)
        q_flat = self.forward(
            state_rep.reshape(B * K, -1),
            candidate_item_vecs.reshape(B * K, -1)
        )
        q_all = q_flat.reshape(B, K)
        if candidate_mask is not None:
            q_all = q_all.masked_fill(~candidate_mask, float("-inf"))
        return q_all



def create_q_networks(state_dim: int,
                    item_dim: int,
                    hidden_dim: int = 256,
                    device: str = "cuda") -> Tuple[QNetwork, QNetwork]:
    q = QNetwork(state_dim, item_dim, hidden_dim).to(device)
    q_target = QNetwork(state_dim, item_dim, hidden_dim).to(device)
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
