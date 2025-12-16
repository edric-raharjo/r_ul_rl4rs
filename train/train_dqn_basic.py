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
        state = batch["state"].to(cfg.device)
        action_item_vec = batch["item_vec"].to(cfg.device)
        reward = batch["reward"].to(cfg.device)
        done = batch["done"].to(cfg.device)
        next_state = batch["next_state"].to(cfg.device)

        next_cand_vecs = batch["next_candidate_item_vecs"].to(cfg.device)
        next_cand_mask = batch["next_candidate_mask"].to(cfg.device).bool()

        q_sa = q(state, action_item_vec)

        with torch.no_grad():
            q_next_all = q_target.q_values_for_candidates(
                state=next_state,
                candidate_item_vecs=next_cand_vecs,
                candidate_mask=next_cand_mask
            )
            q_next_max, _ = q_next_all.max(dim=1)

        loss = criterion(q_sa=q_sa, r=reward, done=done, q_next_max=q_next_max)

        if cfg.do_ascent:
            loss = -loss  # maximize the Q-values

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

def decremental_unlearn(q, forget_loader: DataLoader, cfg: TrainConfig):
    """
    Reinforcement Unlearning dengan random action exploration:
    
    1. Agent pilih action RANDOM dari kandidat yang valid.
    2. Compute Q_random = Q(s, a_random).
    3. Compute Q_policy = Q_ref(s, a_greedy) dari policy lama (greedy).
    4. Update Q dengan loss: |Q_random| + alpha * |Q_random - Q_policy|.
    
    Paper: "Reinforcement Unlearning" by Dayong Ye et al.
    """
    import copy
    from model.loss_function import RandomPolicyUnlearnLoss
    from torch.optim import Adam

    device = cfg.device

    # frozen reference model (policy lama)
    q_ref = copy.deepcopy(q).to(device)
    q_ref.eval()
    for p in q_ref.parameters():
        p.requires_grad_(False)

    q.train()
    for p in q.parameters():
        p.requires_grad_(True)

    unlearn_opt = Adam(q.parameters(), lr=cfg.dec_lr)
    unlearn_loss = RandomPolicyUnlearnLoss(alpha=cfg.dec_alpha, reduction="mean")

    global_step = 0
    for epoch in range(cfg.dec_epochs):
        running = 0.0
        
        for i, batch in enumerate(forget_loader):
            state = batch["state"].to(device)
            cand_vecs = batch["next_candidate_item_vecs"].to(device)
            cand_mask = batch["next_candidate_mask"].to(device).bool()

            B, K, D = cand_vecs.shape

            # ===== STEP 1: RANDOM ACTION =====
            random_actions = []
            for b in range(B):
                valid_indices = torch.where(cand_mask[b])[0]
                if len(valid_indices) > 0:
                    rand_idx = valid_indices[torch.randint(len(valid_indices), (1,))].item()
                else:
                    rand_idx = 0
                random_actions.append(rand_idx)
            
            random_actions = torch.tensor(random_actions, device=device)  # [B]

            # ===== STEP 2: Q_random (model sekarang, action random) =====
            random_action_vecs = cand_vecs[torch.arange(B), random_actions, :]  # [B, D]
            q_random = q(state, random_action_vecs)  # [B]

            # ===== STEP 3: Q_policy (policy lama, action greedy) =====
            with torch.no_grad():
                # Compute Q untuk semua kandidat pakai policy lama
                q_ref_all_list = []
                for k in range(K):
                    q_ref_all_list.append(q_ref(state, cand_vecs[:, k, :]))
                q_ref_all = torch.stack(q_ref_all_list, dim=1)  # [B, K]
                q_ref_all = torch.where(cand_mask, q_ref_all, torch.full_like(q_ref_all, float("-inf")))
                
                # Greedy action dari policy lama
                greedy_actions = q_ref_all.argmax(dim=1)  # [B]
                greedy_action_vecs = cand_vecs[torch.arange(B), greedy_actions, :]  # [B, D]
                q_policy = q_ref(state, greedy_action_vecs)  # [B]

            # ===== STEP 4: LOSS =====
            loss, stats = unlearn_loss(q_random=q_random, q_policy=q_policy)

            if global_step == 0:
                print(f"[DEBUG] Random actions: {random_actions[:5].tolist()}")
                print(f"[DEBUG] Greedy actions (ref): {greedy_actions[:5].tolist()}")
                print(f"[DEBUG] Q_random: {q_random[:5].tolist()}")
                print(f"[DEBUG] Q_policy: {q_policy[:5].tolist()}")
                print(f"[DEBUG] loss.requires_grad={loss.requires_grad}")

            unlearn_opt.zero_grad()
            loss.backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip)

            unlearn_opt.step()

            running += float(loss.item())
            if (global_step % cfg.dec_log_interval) == 0:
                avg = running / max(1, (i + 1))
                print(f"[DEC epoch {epoch+1}/{cfg.dec_epochs} step {global_step}] "
                      f"loss={stats['loss_total']:.6f} avg={avg:.6f} | "
                      f"mag={stats['loss_magnitude']:.6f} diff={stats['loss_diff']:.6f}")
            global_step += 1

    return q


def train_dqn_basic(train_loader: DataLoader,
                    state_dim: int,
                    item_dim: int,
                    hidden_dim: int = 256,
                    cfg: TrainConfig = TrainConfig(),
                    forget_loader: Optional[DataLoader] = None):
    # networks
    q, q_target = create_q_networks(state_dim=state_dim, item_dim=item_dim,
                                    hidden_dim=hidden_dim, device=cfg.device)

    optimizer = Adam(q.parameters(), lr=cfg.lr)
    criterion = DQNLoss(gamma=cfg.gamma, loss_type="huber", reduction="mean")

    global_step = 0
    for epoch in range(cfg.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{cfg.num_epochs} ===")
        global_step = train_one_epoch(q, q_target, train_loader, optimizer, criterion, cfg, global_step)

    # save base model
    save_model(q, cfg.save_dir, cfg.save_name)
    print(f"Saved model to {os.path.join(cfg.save_dir, cfg.save_name)}")

    # optional decremental unlearning
    if cfg.do_decremental:
        if forget_loader is None:
            raise ValueError("cfg.do_decremental=True but forget_loader is None")

        print("\n=== Decremental RL (unlearning) on forget_loader ===")
        q = decremental_unlearn(q, forget_loader, cfg)

        save_model(q, cfg.save_dir, cfg.dec_save_name)
        print(f"Saved decremental model to {os.path.join(cfg.save_dir, cfg.dec_save_name)}")
    
    elif cfg.do_ascent:
        if forget_loader is None:
            raise ValueError("cfg.do_ascent=True but forget_loader is None")

        for epoch in range(cfg.asc_epochs):
            print(f"\n=== Ascent RL (unlearning) Epoch {epoch+1}/{cfg.dec_epochs} on forget_loader ===")
            global_step = train_one_epoch(q, q_target, forget_loader, optimizer, criterion, cfg, global_step)

        save_model(q, cfg.save_dir, cfg.dec_save_name)
        print(f"Saved ascent model to {os.path.join(cfg.save_dir, cfg.dec_save_name)}")

    return q
