# eval/evaluate.py
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from model.dqn_model import QNetwork
from eval.metrics import RecommenderMetrics, UnlearningMetrics
from tqdm import tqdm


def evaluate_recommender(
    q_network: QNetwork,
    dataloader: DataLoader,
    candidate_items: torch.Tensor,
    device: str = 'cuda',
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Comprehensive evaluation of recommender system.
    
    Args:
        q_network: Q-network to evaluate
        dataloader: DataLoader with test data
        candidate_items: All candidate item features [num_items, action_dim]
        device: Device
        k_values: List of K values for top-K metrics
    
    Returns:
        Dictionary of metrics
    """
    q_network.eval()
    candidate_items = candidate_items.to(device)
    num_items = len(candidate_items)
    
    all_q_pred = []
    all_rewards = []
    all_target_indices = []
    all_q_values_all_items = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            states, actions, rewards, next_states, dones = batch
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            
            batch_size = len(states)
            
            # Q-values for actual actions
            q_pred = q_network(states, actions)
            all_q_pred.append(q_pred.cpu())
            all_rewards.append(rewards.cpu())
            
            # Q-values for all candidate items (for ranking metrics)
            q_all_items = []
            for i in range(batch_size):
                state_i = states[i:i+1].expand(num_items, -1)  # [num_items, state_dim]
                q_vals = q_network(state_i, candidate_items)  # [num_items]
                q_all_items.append(q_vals)
            
            q_all_items = torch.stack(q_all_items, dim=0)  # [batch_size, num_items]
            all_q_values_all_items.append(q_all_items.cpu())
            
            # Find target item indices (which item was actually chosen)
            # Match action features to candidate_items to find index
            target_idx = []
            for i in range(batch_size):
                action_i = actions[i]
                # Find matching candidate (assuming first feature is item_idx)
                item_idx_in_action = int(action_i[0].item())
                target_idx.append(item_idx_in_action)
            all_target_indices.append(torch.tensor(target_idx))
    
    # Concatenate all batches
    all_q_pred = torch.cat(all_q_pred, dim=0)
    all_rewards = torch.cat(all_rewards, dim=0)
    all_q_values_all_items = torch.cat(all_q_values_all_items, dim=0)
    all_target_indices = torch.cat(all_target_indices, dim=0)
    
    # Compute metrics
    metrics = RecommenderMetrics()
    results = {
        'mse': metrics.q_value_mse(all_q_pred, all_rewards),
        'mae': metrics.q_value_mae(all_q_pred, all_rewards),
        'mrr': metrics.mrr(all_q_values_all_items, all_target_indices)
    }
    
    # Top-K metrics
    for k in k_values:
        results[f'hit_rate@{k}'] = metrics.hit_rate(all_q_values_all_items, all_target_indices, k)
        results[f'ndcg@{k}'] = metrics.ndcg_at_k(all_q_values_all_items, all_target_indices, k)
        results[f'precision@{k}'] = metrics.precision_at_k(all_q_values_all_items, all_target_indices, k)
    
    q_network.train()
    return results


def evaluate_unlearning(
    results_baseline: Dict[str, float],
    results_after_learning: Dict[str, float],
    results_after_unlearning: Dict[str, float],
    metric_name: str = 'hit_rate@10'
) -> Dict[str, float]:
    """
    Evaluate unlearning effectiveness.
    
    Args:
        results_baseline: Metrics on forget set from baseline model (trained on retain only)
        results_after_learning: Metrics on forget set after learning on forget data
        results_after_unlearning: Metrics on forget set after unlearning
        metric_name: Primary metric to use for evaluation
    
    Returns:
        Dictionary with unlearning-specific metrics
    """
    unlearn_metrics = UnlearningMetrics()
    
    # Forget quality: how much did performance drop on forget set?
    forget_quality = unlearn_metrics.forget_quality(
        forget_performance_after=results_after_unlearning[metric_name],
        forget_performance_before=results_after_learning[metric_name]
    )
    
    # Model utility is computed on retain set (passed separately)
    # This function just returns forget quality
    return {
        'forget_quality': forget_quality,
        'forget_performance_baseline': results_baseline[metric_name],
        'forget_performance_learned': results_after_learning[metric_name],
        'forget_performance_unlearned': results_after_unlearning[metric_name]
    }


def print_evaluation_results(
    phase_name: str,
    test_results: Dict[str, float],
    forget_results: Dict[str, float]
):
    """
    Pretty print evaluation results.
    
    Args:
        phase_name: Name of the phase (e.g., "Baseline", "After Learning")
        test_results: Metrics on test set
        forget_results: Metrics on forget set
    """
    print(f"\n{'='*60}")
    print(f"{phase_name} - Evaluation Results")
    print(f"{'='*60}")
    
    print("\n--- Test Set ---")
    for metric, value in test_results.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    print("\n--- Forget Set ---")
    for metric, value in forget_results.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    print(f"{'='*60}\n")


def compute_unlearning_summary(
    test_baseline: Dict[str, float],
    test_learned: Dict[str, float],
    test_unlearned: Dict[str, float],
    forget_baseline: Dict[str, float],
    forget_learned: Dict[str, float],
    forget_unlearned: Dict[str, float],
    primary_metric: str = 'hit_rate@10'
) -> Dict[str, float]:
    """
    Compute comprehensive unlearning summary metrics.
    
    Returns:
        Summary dictionary with key unlearning metrics
    """
    unlearn_metrics = UnlearningMetrics()
    
    # Forget quality
    forget_quality = unlearn_metrics.forget_quality(
        forget_performance_after=forget_unlearned[primary_metric],
        forget_performance_before=forget_learned[primary_metric]
    )
    
    # Model utility (on test set)
    model_utility = unlearn_metrics.model_utility(
        retain_performance_after=test_unlearned[primary_metric],
        retain_performance_baseline=test_baseline[primary_metric]
    )
    
    # Overall efficacy
    efficacy = unlearn_metrics.unlearning_efficacy(forget_quality, model_utility)
    
    summary = {
        'forget_quality': forget_quality,
        'model_utility': model_utility,
        'unlearning_efficacy': efficacy,
        f'test_{primary_metric}_baseline': test_baseline[primary_metric],
        f'test_{primary_metric}_learned': test_learned[primary_metric],
        f'test_{primary_metric}_unlearned': test_unlearned[primary_metric],
        f'forget_{primary_metric}_baseline': forget_baseline[primary_metric],
        f'forget_{primary_metric}_learned': forget_learned[primary_metric],
        f'forget_{primary_metric}_unlearned': forget_unlearned[primary_metric]
    }
    
    return summary


def print_unlearning_summary(summary: Dict[str, float]):
    """Pretty print unlearning summary."""
    print(f"\n{'='*60}")
    print("UNLEARNING SUMMARY")
    print(f"{'='*60}")
    print(f"  Forget Quality:       {summary['forget_quality']:.4f}  (higher = better forgetting)")
    print(f"  Model Utility:        {summary['model_utility']:.4f}  (higher = better retention)")
    print(f"  Unlearning Efficacy:  {summary['unlearning_efficacy']:.4f}  (harmonic mean)")
    print(f"{'='*60}\n")
