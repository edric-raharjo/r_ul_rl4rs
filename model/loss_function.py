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
