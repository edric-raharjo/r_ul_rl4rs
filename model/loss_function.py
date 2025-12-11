# model/loss_function.py
from typing import Literal

import torch
import torch.nn as nn


class DQNLoss(nn.Module):
    """
    Loss DQN standar:
      target = r + gamma * (1 - done) * max_a' Q_target(s', a')
      L = MSE(Q(s,a), target)
    """
    def __init__(self,
                 gamma: float = 0.99,
                 reduction: Literal["mean", "sum"] = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self,
                q_sa: torch.Tensor,           # [B]
                r: torch.Tensor,              # [B]
                done: torch.Tensor,           # [B], 1 jika terminal
                q_next_max: torch.Tensor      # [B], max_a' Q_target(s',a')
                ) -> torch.Tensor:
        """
        Hitung DQN loss untuk 1 batch.
        """
        # y = r + gamma * (1 - done) * q_next_max
        target = r + self.gamma * (1.0 - done) * q_next_max
        target = target.detach()
        loss = self.mse(q_sa, target)
        return loss


# TODO:
""" 
a. Decremental RL
b. Env Poisoning
c. Inverse Loss Function
"""