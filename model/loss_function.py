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
from typing import Literal
import torch
import torch.nn as nn

class RandomPolicyUnlearnLoss(nn.Module):
    """
    Loss untuk unlearning dengan random action exploration:
    
    L = E[ |Q_random(s, a_rand)| ] + alpha * E[ |Q_random(s, a_rand) - Q_policy(s, a_greedy)| ]
    
    - Q_random: Q dari model yang sedang di-unlearn untuk action random.
    - Q_policy: Q dari model reference (policy sebelum unlearning) untuk action greedy.
    - alpha: weight untuk regularizer (jaga jangan terlalu jauh dari policy lama).
    """
    
    def __init__(self, alpha: float = 1.0, reduction: Literal["mean", "sum"] = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.reduction = reduction
    
    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return x.mean()
        elif self.reduction == "sum":
            return x.sum()
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")
    
    def forward(self,
                q_random: torch.Tensor,      # [B] Q untuk action random
                q_policy: torch.Tensor       # [B] Q untuk action greedy dari policy lama
                ) -> tuple[torch.Tensor, dict]:
        
        q_random = q_random.float()
        q_policy = q_policy.float().detach()  # frozen
        
        # Term 1: minimize magnitude Q random action
        loss_mag = self._reduce(torch.abs(q_random))
        
        # Term 2: minimize difference (regularizer)
        loss_diff = self._reduce(torch.abs(q_random - q_policy))
        
        loss_total = loss_mag + self.alpha * loss_diff
        
        stats = {
            "loss_total": float(loss_total.detach().cpu()),
            "loss_magnitude": float(loss_mag.detach().cpu()),
            "loss_diff": float(loss_diff.detach().cpu()),
            "alpha": float(self.alpha),
        }
        return loss_total, stats
