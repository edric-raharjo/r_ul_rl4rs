# model/loss_function.py
from typing import Literal, Optional

import torch
import torch.nn as nn

class DQNLoss(nn.Module):
    """
    Basic DQN loss:
      y = r + gamma * (1 - done) * max_a' Q_target(s', a')
      L = loss_fn(Q(s,a), y)

    Notes:
    - done harus 0/1 float
    - q_next_max harus sudah respect mask (invalid action -> -inf)
    """
    def __init__(self,
                 gamma: float = 0.99,
                 loss_type: Literal["mse", "huber"] = "huber",
                 reduction: Literal["mean", "sum"] = "mean",
                 clamp_q_next: bool = True):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.clamp_q_next = clamp_q_next

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == "huber":
            self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
        else:
            raise ValueError("loss_type must be 'mse' or 'huber'")

    def forward(self,
                q_sa: torch.Tensor,       # model output untuk action yang diambil
                r: torch.Tensor,          # reward dari dataset
                done: torch.Tensor,       # 0/1
                q_next_max: torch.Tensor  # max_a' Q_target(s',a') setelah masking
                ) -> torch.Tensor:

        # Pastikan tipe float
        r = r.float()
        done = done.float()
        q_sa = q_sa.float()
        q_next_max = q_next_max.float()

        # Kalau semua action invalid, q_next_max bisa -inf.
        # Untuk basic DQN, cara aman: treat jadi 0 (seolah tidak ada future value).
        if self.clamp_q_next:
            q_next_max = torch.where(torch.isfinite(q_next_max),
                                     q_next_max,
                                     torch.zeros_like(q_next_max))

        # Target Bellman (tanpa grad)
        with torch.no_grad():
            target = r + self.gamma * (1.0 - done) * q_next_max

        # Loss
        return self.loss_fn(q_sa, target)

# TODO:
""" 
a. Decremental RL
"""

import torch
import torch.nn as nn
from typing import Literal, Dict, Tuple


class RandomPolicyUnlearnLoss(nn.Module):
    """
    Random-Policy Decremental RL loss (Ye et al., Reinforcement Unlearning).

    Inputs:
        q_random: Q_cur(s, a_rand) from CURRENT model on random valid actions [B]
        q_policy: Q_ref(s, a_greedy_ref) from FROZEN reference model on its greedy actions [B]

    Loss (per batch):
        L_mag  = E[ |q_random| ]
        L_diff = E[ |q_random - q_policy| ]
        L_tot  = L_mag + alpha * L_diff

    - L_mag  pushes Q on random actions in forget states toward 0
    - L_diff keeps the new Q surface close to the old policy's Q on those states
    """

    def __init__(
        self,
        alpha: float = 0.5,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        q_random: torch.Tensor,   # [B]
        q_policy: torch.Tensor,   # [B]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        q_random = q_random.float()
        q_policy = q_policy.float()

        # magnitude term: make random actions low-value
        loss_mag = torch.abs(q_random)

        # consistency term: keep close to reference policy Q
        loss_diff = torch.abs(q_random - q_policy)

        if self.reduction == "mean":
            loss_mag = loss_mag.mean()
            loss_diff = loss_diff.mean()
        elif self.reduction == "sum":
            loss_mag = loss_mag.sum()
            loss_diff = loss_diff.sum()
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")

        loss_total = loss_mag + self.alpha * loss_diff

        stats = {
            "loss_total": float(loss_total.detach().cpu().item()),
            "loss_magnitude": float(loss_mag.detach().cpu().item()),
            "loss_diff": float(loss_diff.detach().cpu().item()),
        }

        return loss_total, stats

