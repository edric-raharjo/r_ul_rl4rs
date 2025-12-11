# model/dqn_model.py
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    """
    Encode:
      - history embedding (sudah kamu siapkan di Dataset sebagai vektor)
      - tier/status flags (optional)
    Input: state_tensor [B, state_dim]
    Output: state_emb   [B, hidden_dim]
    """
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return x


class QNetwork(nn.Module):
    """
    Parametric DQN:
      Q(s,a) = f([state_emb || item_vec])
    - state: [B, state_dim]
    - item_vec: [B, item_dim]
    Output:
      - q_sa: [B]
    """
    def __init__(self,
                 state_dim: int,
                 item_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        self.state_encoder = StateEncoder(state_dim, hidden_dim)

        self.fc_q1 = nn.Linear(hidden_dim + item_dim, hidden_dim)
        self.fc_q2 = nn.Linear(hidden_dim, 1)

    def forward(self,
                state: torch.Tensor,
                item_vec: torch.Tensor) -> torch.Tensor:
        """
        state:    [B, state_dim]
        item_vec: [B, item_dim]
        """
        s_emb = self.state_encoder(state)                  # [B, H]
        x = torch.cat([s_emb, item_vec], dim=-1)           # [B, H+item_dim]
        x = F.relu(self.fc_q1(x))
        q = self.fc_q2(x).squeeze(-1)                      # [B]
        return q


# ---------- Helper untuk target-network & save/load ----------

def create_q_networks(state_dim: int,
                      item_dim: int,
                      hidden_dim: int = 256,
                      device: str = "cuda") -> Tuple[QNetwork, QNetwork]:
    q = QNetwork(state_dim, item_dim, hidden_dim).to(device)
    q_target = QNetwork(state_dim, item_dim, hidden_dim).to(device)
    q_target.load_state_dict(q.state_dict())
    q_target.eval()
    return q, q_target


def soft_update(target_net: nn.Module,
                source_net: nn.Module,
                tau: float = 0.005):
    """
    Soft update: target = tau * source + (1-tau) * target
    """
    for target_param, param in zip(target_net.parameters(),
                                   source_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1.0 - tau) * target_param.data)


def hard_update(target_net: nn.Module,
                source_net: nn.Module):
    """
    Hard update: target <- source
    """
    target_net.load_state_dict(source_net.state_dict())


# ---------- Save / load weights ----------

def save_model(model: nn.Module,
               save_dir: str,
               filename: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module,
               save_dir: str,
               filename: str,
               map_location: str = "cpu"):
    path = os.path.join(save_dir, filename)
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
