import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.dqn_model import create_q_networks, hard_update, soft_update, save_model
from model.loss_function import DQNLoss # <-- tambah


@dataclass
class TrainConfig:
    device: str = "cuda"
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 256
    num_epochs: int = 3
    grad_clip: float = 5.0

    # target network update
    target_update: str = "hard"      # "hard" or "soft"
    hard_update_interval: int = 200  # steps
    tau: float = 0.005               # for soft update

    # logging/saving
    log_interval: int = 50
    save_dir: str = "weights"
    save_name: str = "dqn_basic.pt"

    # ---- decremental RL (unlearning) ----
    do_decremental: bool = False
    dec_epochs: int = 3
    dec_lr: float = 1e-4
    dec_alpha: float = 0.5          # alpha pada Eq(5)
    dec_log_interval: int = 50
    dec_save_name: str = "dqn_decremental.pt"

    # ---- ascent RL (unlearning) ----
    do_ascent: bool = False
    asc_epochs: int = 3

def train_one_epoch(q, q_target, loader, optimizer, criterion, cfg: TrainConfig, global_step: int):
    q.train()
    running_loss = 0.0

    for i, batch in enumerate(loader):
        state = batch["state"].to(cfg.device)                 # [B, state_dim]
        action_id = batch["action_id"].to(cfg.device).long()  # [B]
        reward = batch["reward"].to(cfg.device)
        done = batch["done"].to(cfg.device)
        next_state = batch["next_state"].to(cfg.device)

        next_cand_ids = batch["next_candidate_ids"].to(cfg.device).long()        # [B, K]
        next_cand_mask = batch["next_candidate_mask"].to(cfg.device).bool()      # [B, K]

        # current Q(s,a)
        q_all = q(state)                             # [B, num_items]
        q_sa = q_all.gather(1, action_id.unsqueeze(1)).squeeze(1)  # [B]

        # target Q on next state
        with torch.no_grad():
            q_next_all = q_target.q_values_for_candidates(
                state=next_state,
                candidate_ids=next_cand_ids,
                candidate_mask=next_cand_mask,
            )                                        # [B, K]
            q_next_max, _ = q_next_all.max(dim=1)    # [B]

        loss = criterion(q_sa=q_sa, r=reward, done=done, q_next_max=q_next_max)

        if cfg.do_ascent:
            loss = -loss

        optimizer.zero_grad()
        loss.backward()

        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip)

        optimizer.step()

        # target update
        if cfg.target_update == "hard":
            if global_step % cfg.hard_update_interval == 0:
                hard_update(q_target, q)
        elif cfg.target_update == "soft":
            soft_update(q_target, q, tau=cfg.tau)
        else:
            raise ValueError("target_update must be 'hard' or 'soft'")

        running_loss += float(loss.item())
        if (global_step % cfg.log_interval) == 0:
            avg = running_loss / max(1, (i + 1))
            print(f"[step {global_step}] loss={loss.item():.6f} avg_loss={avg:.6f}")

        global_step += 1

    return global_step

def train_dqn_basic(train_loader: DataLoader,
                    state_dim: int,
                    num_items: int,                 # <-- was item_dim
                    hidden_dim: int = 256,
                    cfg: TrainConfig = TrainConfig(),
                    forget_loader: Optional[DataLoader] = None,
                    retain_loader: Optional[DataLoader] = None):
    # networks
    q, q_target = create_q_networks(
        state_dim=state_dim,
        num_items=num_items,    
        hidden_dim=hidden_dim,
        device=cfg.device,
    )

    optimizer = Adam(q.parameters(), lr=cfg.lr)
    criterion = DQNLoss(gamma=cfg.gamma, loss_type="huber", reduction="mean")

    # handle ascent flag
    enable_later = False
    if cfg.do_ascent:
        cfg.do_ascent = False  # disable during base training
        enable_later = True

    global_step = 0
    for epoch in range(cfg.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{cfg.num_epochs} ===")
        global_step = train_one_epoch(
            q, q_target, train_loader, optimizer, criterion, cfg, global_step
        )

    # save base model
    save_model(q, cfg.save_dir, cfg.save_name)
    print(f"Saved model to {os.path.join(cfg.save_dir, cfg.save_name)}")

    # optional decremental unlearning
    if cfg.do_decremental:
        if forget_loader is None or retain_loader is None:
            raise ValueError("Need both forget_loader and retain_loader for decremental RL")

        print("\n=== Decremental RL (Eq.12-13) on forget/retain loaders ===")
        q = decremental_unlearn_eq12_13(q, forget_loader, retain_loader, cfg, lambda_retain=cfg.dec_alpha)

    elif enable_later:
        # re-enable ascent only after base training
        cfg.do_ascent = True

        if forget_loader is None:
            raise ValueError("cfg.do_ascent=True but forget_loader is None")

        for epoch in range(cfg.asc_epochs):
            print(f"\n=== Ascent RL (unlearning) Epoch {epoch+1}/{cfg.asc_epochs} on forget_loader ===")
            global_step = train_one_epoch(
                q, q_target, forget_loader, optimizer, criterion, cfg, global_step
            )

        save_model(q, cfg.save_dir, cfg.dec_save_name)
        print(f"Saved ascent model to {os.path.join(cfg.save_dir, cfg.dec_save_name)}")

    return q

import copy
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam


def decremental_unlearn_eq12_13(
    q,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    cfg: TrainConfig,
    lambda_retain: float = 1.0,
):
    """
    Decremental RL (Ye et al.) for catalog-style DQN, implementing Eq. 12 & 13.

    - Forget loss (Eq. 12):   L_forget  = E_{(s,a)~tau_u} [ Q_cur(s,a) ]
    - Retain loss (Eq. 13):   L_retain  = E_{(s,a) !~tau_u} | Q_cur(s,a) - Q_ref(s,a) |
    - Total:                  L_dec     = L_forget + lambda_retain * L_retain
    """

    device = cfg.device

    # ---- 1. Frozen reference network Q_ref = Q_theta ----
    q_ref = copy.deepcopy(q).to(device)
    q_ref.eval()
    for p in q_ref.parameters():
        p.requires_grad_(False)

    # ---- 2. Trainable network Q_cur = Q_theta' ----
    q.train()
    for p in q.parameters():
        p.requires_grad_(True)

    opt = Adam(q.parameters(), lr=cfg.dec_lr)

    # simple iterator cycling over retain_loader
    retain_iter = iter(retain_loader)

    global_step = 0
    for epoch in range(cfg.dec_epochs):
        running = 0.0

        for i, batch_f in enumerate(forget_loader):
            # -------------------------
            # FORGET BATCH  (Eq. 12)
            # -------------------------
            s_f = batch_f["state"].to(device)          # [B_f, state_dim]
            a_f = batch_f["action_id"].to(device).long()  # [B_f]

            q_all_f = q(s_f)                           # [B_f, num_items]
            q_forget = q_all_f.gather(1, a_f.unsqueeze(1)).squeeze(1)  # [B_f]

            # Eq. 12: minimize reward along tau_u -> push Q down
            L_forget = q_forget.mean()

            # -------------------------
            # RETAIN BATCH (Eq. 13)
            # -------------------------
            try:
                batch_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                batch_r = next(retain_iter)

            s_r = batch_r["state"].to(device)          # [B_r, state_dim]
            a_r = batch_r["action_id"].to(device).long()

            # current Q on retain data
            q_all_r_cur = q(s_r)                       # [B_r, num_items]
            q_retain_cur = q_all_r_cur.gather(1, a_r.unsqueeze(1)).squeeze(1)  # [B_r]

            # reference Q on retain data (no grad)
            with torch.no_grad():
                q_all_r_ref = q_ref(s_r)               # [B_r, num_items]
                q_retain_ref = q_all_r_ref.gather(1, a_r.unsqueeze(1)).squeeze(1)  # [B_r]

            # Eq. 13: keep new Q close to old Q on retain trajectories
            L_retain = torch.abs(q_retain_cur - q_retain_ref).mean()

            # -------------------------
            # TOTAL DECREMENTAL LOSS
            # -------------------------
            loss = L_forget + lambda_retain * L_retain

            opt.zero_grad()
            loss.backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip)

            opt.step()

            running += float(loss.item())
            if (global_step % cfg.dec_log_interval) == 0:
                avg = running / max(1, (i + 1))
                print(
                    f"[DEC-EQ12_13 epoch {epoch+1}/{cfg.dec_epochs} "
                    f"step {global_step}] loss={loss.item():.6f} avg={avg:.6f} "
                    f"| L_forget={L_forget.item():.6f} L_retain={L_retain.item():.6f}"
                )

            global_step += 1

    return q
