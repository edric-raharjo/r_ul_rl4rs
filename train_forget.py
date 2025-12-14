import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data_loader import LogConfig, ItemConfig, load_log_table, load_item_table
from dataset.rl4rs_dataset import RL4RSDataset9Step
from train.train_dqn_basic import train_dqn_basic, TrainConfig
from eval.evaluate import evaluate


def split_df_log_train_forget_test(df_log, train_ratio=0.6, forget_ratio=0.2, test_ratio=0.2, seed=42):
    assert abs(train_ratio + forget_ratio + test_ratio - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df_log))
    rng.shuffle(idx)

    n_train = int(len(df_log) * train_ratio)
    n_forget = int(len(df_log) * forget_ratio)

    train_idx = idx[:n_train]
    forget_idx = idx[n_train:n_train + n_forget]
    test_idx = idx[n_train + n_forget:]

    return (
        df_log.iloc[train_idx].reset_index(drop=True),
        df_log.iloc[forget_idx].reset_index(drop=True),
        df_log.iloc[test_idx].reset_index(drop=True),
    )


def make_dataset(df_log_part, df_item):
    return RL4RSDataset9Step(
        df_log=df_log_part,
        df_item=df_item,
        slate_size=9,
        use_tier_flags=True,
        tier_weights=(1.0, 2.0, 4.0),
        exclude_history_candidates=True
    )


def train_on(df_log_train, df_item, cfg, forget_loader=None):
    ds = make_dataset(df_log_train, df_item)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    sample0 = ds[0]
    state_dim = sample0["state"].numel()
    item_dim = sample0["item_vec"].numel()

    q = train_dqn_basic(
        train_loader=loader,
        state_dim=state_dim,
        item_dim=item_dim,
        hidden_dim=256,
        cfg=cfg,
        forget_loader=forget_loader  # <-- NEW: untuk decremental RL
    )
    return q


def eval_report(tag, model, test_ds, forget_ds, device, ks=(1, 3, 5, 9)):
    print(f"\n=== {tag} ===")
    print("Test:", evaluate(model, test_ds, device=device, ks=ks))
    print("Forget:", evaluate(model, forget_ds, device=device, ks=ks))


def main():
    EPOCH = 3
    log_path = r"E:\Kuliah\Kuliah\Kuliah\PRODI\Semester 7\ProSkripCode\data\raw\trainset.csv"
    item_path = r"E:\Kuliah\Kuliah\Kuliah\PRODI\Semester 7\ProSkripCode\data\raw\item_info.csv"

    log_cfg = LogConfig(path=log_path, slate_size=9, max_click_history=50)
    item_cfg = ItemConfig(path=item_path, item_vec_dim=None)

    df_log = load_log_table(log_cfg).reset_index(drop=True)
    df_item = load_item_table(item_cfg)

    df_log = df_log.iloc[:200].reset_index(drop=True)

    df_train, df_forget, df_test = split_df_log_train_forget_test(df_log, 0.6, 0.2, 0.2, seed=42)

    test_ds = make_dataset(df_test, df_item)
    forget_ds = make_dataset(df_forget, df_item)

    # Loader untuk forget (dipakai saat decremental/unlearning)
    forget_loader = DataLoader(
        forget_ds,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    cfg = TrainConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=1e-3,
        gamma=0.99,
        batch_size=256,
        num_epochs=EPOCH,
        target_update="hard",
        hard_update_interval=500,
        save_dir="weights",
        save_name="dqn_basic.pt"
    )

    # A: untrained baseline (optional)
    # try:
    #     cfg_A = TrainConfig(
    #         device=cfg.device, lr=cfg.lr, gamma=cfg.gamma, batch_size=cfg.batch_size,
    #         num_epochs=0,
    #         target_update=cfg.target_update, hard_update_interval=cfg.hard_update_interval,
    #         save_dir=cfg.save_dir, save_name="dqn_untrained.pt"
    #     )
    #     q_A = train_on(df_train, df_item, cfg_A)
    #     eval_report("A (untrained)", q_A, test_ds, forget_ds, cfg.device)
    # except Exception as e:
    #     print("\n(A untrained diskip) train_dqn_basic tidak support num_epochs=0:", repr(e))

    # B: trained (basic)
    q_B = train_on(df_train, df_item, cfg)
    eval_report("B (trained)", q_B, test_ds, forget_ds, cfg.device)

    # C: trained + decremental RL unlearning on forget
    cfg_C = TrainConfig(
        device=cfg.device,
        lr=cfg.lr,
        gamma=cfg.gamma,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        target_update=cfg.target_update,
        hard_update_interval=cfg.hard_update_interval,
        save_dir=cfg.save_dir,
        save_name="dqn_basic_then_dec.pt"
    )

    # --- aktifkan decremental ---
    cfg_C.do_decremental = True
    cfg_C.dec_epochs = EPOCH
    cfg_C.dec_lr = 1e-3
    cfg_C.dec_alpha = 0.45
    cfg_C.dec_save_name = "dqn_decremental.pt"

    q_C = train_on(df_train, df_item, cfg_C, forget_loader=forget_loader)
    eval_report("C (basic + decremental)", q_C, test_ds, forget_ds, cfg.device)

    @torch.no_grad()
    def compute_avg_q_forget(model, forget_ds, device):
        loader = DataLoader(forget_ds, batch_size=256, shuffle=False)
        q_vals = []
        for batch in loader:
            state = batch["state"].to(device)
            cand_vecs = batch["next_candidate_item_vecs"].to(device)
            cand_mask = batch["next_candidate_mask"].to(device).bool()
            # manual forward (atau pakai q_values_for_candidates kalau sudah fix grad)
            B, K, D = cand_vecs.shape
            qs = []
            for k in range(K):
                qs.append(model(state, cand_vecs[:, k, :]))
            q_vec = torch.stack(qs, dim=1)
            q_vec = torch.where(cand_mask, q_vec, torch.zeros_like(q_vec))
            q_vals.append(q_vec.mean().item())
        return float(np.mean(q_vals))

    # RETAIN SET TESTING
    # Sebelum decremental:
    avg_q_test_before = compute_avg_q_forget(q_B, test_ds, cfg.device)
    # Setelah decremental:
    avg_q_test_after = compute_avg_q_forget(q_C, test_ds, cfg.device)
    print(f"Avg Q retain: before={avg_q_test_before:.4f}, after={avg_q_test_after:.4f}")
    
    # FORGET SET TESTING
    # Sebelum decremental:
    avg_q_before = compute_avg_q_forget(q_B, forget_ds, cfg.device)
    # Setelah decremental:
    avg_q_after = compute_avg_q_forget(q_C, forget_ds, cfg.device)
    print(f"Avg Q forget: before={avg_q_before:.4f}, after={avg_q_after:.4f}")



if __name__ == "__main__":
    main()
