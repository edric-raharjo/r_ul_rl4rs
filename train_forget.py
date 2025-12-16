import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path

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

def train_on(df_log_train, df_item, cfg, forget_loader=None, retain_loader=None):
    ds = make_dataset(df_log_train, df_item)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    sample0 = ds[0]
    state_dim = sample0["state"].numel()
    num_items = ds.num_items  # from RL4RSDataset9Step

    q = train_dqn_basic(
        train_loader=loader,
        state_dim=state_dim,
        num_items=num_items,
        hidden_dim=256,
        cfg=cfg,
        forget_loader=forget_loader,  # for decremental RL
        retain_loader=retain_loader,  # for decremental RL
    )
    return q



def create_results_folder(trial_name):
    """Create a timestamped results folder with subfolders"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"results/{trial_name}_{timestamp}"
    
    base_path = Path(results_folder)
    base_path.mkdir(exist_ok=True)
    
    # Create subfolders
    (base_path / "csv").mkdir(exist_ok=True)
    (base_path / "plots").mkdir(exist_ok=True)
    
    return str(base_path)


def eval_report(tag, model, test_ds, forget_ds, device, ks=(1, 3, 5, 9)):
    """Evaluate and return results as dict for logging"""
    print(f"\n=== {tag} ===")
    test_metrics = evaluate(model, test_ds, device=device, ks=ks)
    forget_metrics = evaluate(model, forget_ds, device=device, ks=ks)
    
    print("Test:", test_metrics)
    print("Forget:", forget_metrics)
    
    return {
        "trial": tag,
        "test": test_metrics,
        "forget": forget_metrics
    }


def flatten_metrics_to_csv_row(trial_name, test_metrics, forget_metrics):
    """Flatten nested metrics dict into a single CSV row"""
    row = {"trial": trial_name}
    
    # Flatten test metrics
    for key, value in test_metrics.items():
        row[f"test_{key}"] = value
    
    # Flatten forget metrics
    for key, value in forget_metrics.items():
        row[f"forget_{key}"] = value
    
    return row


def save_metrics_to_csv(results_folder, metrics_list):
    """Save all metrics to a CSV file"""
    df = pd.DataFrame(metrics_list)
    csv_path = Path(results_folder) / "csv" / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to {csv_path}")
    return csv_path


def create_plots(results_folder, metrics_list):
    """Create and save comparison plots"""
    df = pd.DataFrame(metrics_list)
    
    plots_folder = Path(results_folder) / "plots"
    
    # Extract metrics (assuming NDCG@k, HR@k format)
    trials = df["trial"].tolist()
    
    # Find all metric columns
    test_cols = [col for col in df.columns if col.startswith("test_")]
    forget_cols = [col for col in df.columns if col.startswith("forget_")]
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Evaluation Metrics Comparison", fontsize=16, fontweight='bold')
    
    # Plot 1: Test metrics across trials
    if test_cols:
        ax = axes[0, 0]
        for col in test_cols[:4]:  # Limit to first 4 metrics
            ax.plot(trials, df[col], marker='o', label=col)
        ax.set_title("Test Metrics")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Forget metrics across trials
    if forget_cols:
        ax = axes[0, 1]
        for col in forget_cols[:4]:  # Limit to first 4 metrics
            ax.plot(trials, df[col], marker='s', label=col)
        ax.set_title("Forget Metrics")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Test vs Forget comparison for first metric
    if test_cols and forget_cols:
        ax = axes[1, 0]
        test_metric = test_cols[0]
        forget_metric = forget_cols[0]
        x = np.arange(len(trials))
        width = 0.35
        ax.bar(x - width/2, df[test_metric], width, label='Test', alpha=0.8)
        ax.bar(x + width/2, df[forget_metric], width, label='Forget', alpha=0.8)
        ax.set_title(f"Test vs Forget: {test_metric}")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(trials, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "Summary Statistics:\n\n"
    for col in test_cols[:3]:
        mean_val = df[col].mean()
        std_val = df[col].std()
        summary_text += f"{col}:\n  Mean: {mean_val:.4f}, Std: {std_val:.4f}\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_path = plots_folder / "metrics_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to {plot_path}")
    plt.close()
    
    return plot_path


def main(
    log_path = r"E:\Kuliah\Kuliah\Kuliah\PRODI\Semester 7\ProSkripCode\data\raw\trainset.csv",
    item_path = r"E:\Kuliah\Kuliah\Kuliah\PRODI\Semester 7\ProSkripCode\data\raw\item_info.csv"
        ):
    EPOCH = 2

    log_cfg = LogConfig(path=log_path, slate_size=9, max_click_history=50)
    item_cfg = ItemConfig(path=item_path, item_vec_dim=None)

    df_log = load_log_table(log_cfg).reset_index(drop=True)
    df_item = load_item_table(item_cfg)

    df_log = df_log.iloc[:50].reset_index(drop=True)

    df_train, df_forget, df_test = split_df_log_train_forget_test(df_log, 0.6, 0.2, 0.2, seed=42)
    df_train_forget = pd.concat([df_train, df_forget]).reset_index(drop=True)

    test_ds = make_dataset(df_test, df_item)
    forget_ds = make_dataset(df_forget, df_item)
    train_ds = make_dataset(df_train, df_item)

    # Loader untuk forget (dipakai saat decremental/unlearning)
    forget_loader = DataLoader(
        forget_ds,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    retain_loader = DataLoader(
        train_ds,
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

    # Create results folder
    results_folder = create_results_folder("RL4RS_Unlearning")
    print(f"\n{'='*60}")
    print(f"Results will be saved to: {results_folder}")
    print(f"{'='*60}")
    
    # Store all results
    all_metrics = []

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

    # B: trained on retain only (basic)
    q_B = train_on(df_train, df_item, cfg)
    metrics_B = eval_report("A (trained on retain)", q_B, test_ds, forget_ds, cfg.device)
    all_metrics.append(flatten_metrics_to_csv_row(metrics_B["trial"], metrics_B["test"], metrics_B["forget"]))

    # B_alt: trained on retain + forget (basic)
    cfg.save_name = "dqn_basic_retain_forget.pt"
    q_B_rf = train_on(df_train_forget, df_item, cfg)
    metrics_B_rf = eval_report("B (trained on retain+forget)", q_B_rf, test_ds, forget_ds, cfg.device)
    all_metrics.append(flatten_metrics_to_csv_row(metrics_B_rf["trial"], metrics_B_rf["test"], metrics_B_rf["forget"]))

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
    cfg_C.dec_alpha = 0.5
    cfg_C.dec_save_name = "dqn_decremental.pt"

    q_C = train_on(df_train_forget, df_item, cfg_C, forget_loader=forget_loader, retain_loader=retain_loader)
    metrics_C = eval_report("C (basic + decremental)", q_C, test_ds, forget_ds, cfg.device)
    all_metrics.append(flatten_metrics_to_csv_row(metrics_C["trial"], metrics_C["test"], metrics_C["forget"]))

    # D: Learn with retain and forget, then gradient ascent on forget
    cfg_D = TrainConfig(
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

    cfg_D.do_ascent = True
    cfg_D.asc_epochs = EPOCH

    q_D = train_on(df_train_forget, df_item, cfg_D, forget_loader=forget_loader)
    metrics_D = eval_report("D (retain+forget + ascent)", q_D, test_ds, forget_ds, cfg.device)
    all_metrics.append(flatten_metrics_to_csv_row(metrics_D["trial"], metrics_D["test"], metrics_D["forget"]))

    # Save metrics to CSV
    save_metrics_to_csv(results_folder, all_metrics)
    
    # Create and save plots
    create_plots(results_folder, all_metrics)
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {results_folder}")
    print(f"  - CSV: {results_folder}/csv/metrics.csv")
    print(f"  - Plots: {results_folder}/plots/metrics_comparison.png")
    print(f"{'='*60}")



if __name__ == "__main__":
    main()
