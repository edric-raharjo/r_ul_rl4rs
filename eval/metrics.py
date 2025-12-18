# eval/metrics.py
import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RecommenderMetrics:
    """
    Evaluation metrics for DQN-based recommender system.
    """
    
    @staticmethod
    def hit_rate(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
        """
        Hit Rate@K: Fraction of times the target item appears in top-K predictions.
        
        Args:
            predictions: Predicted Q-values [batch_size, num_items]
            targets: Target item indices [batch_size]
            k: Top-K items to consider
        
        Returns:
            Hit rate as float
        """
        _, top_k_items = torch.topk(predictions, k, dim=1)  # [batch_size, k]
        targets = targets.unsqueeze(1)  # [batch_size, 1]
        hits = (top_k_items == targets).any(dim=1).float()
        return hits.mean().item()
    
    
    @staticmethod
    def ndcg_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain@K.
        
        Args:
            predictions: Predicted Q-values [batch_size, num_items]
            targets: Target item indices [batch_size]
            k: Top-K items to consider
        
        Returns:
            NDCG@K as float
        """
        batch_size = predictions.size(0)
        _, top_k_items = torch.topk(predictions, k, dim=1)  # [batch_size, k]
        
        ndcg_scores = []
        for i in range(batch_size):
            # Find position of target item in top-k
            target_idx = targets[i].item()
            top_k = top_k_items[i].tolist()
            
            if target_idx in top_k:
                position = top_k.index(target_idx) + 1  # 1-indexed
                dcg = 1.0 / np.log2(position + 1)
                idcg = 1.0 / np.log2(2)  # Best case: target at position 1
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
            
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores)
    
    
    @staticmethod
    def precision_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
        """
        Precision@K: Fraction of recommended items that are relevant.
        
        Args:
            predictions: Predicted Q-values [batch_size, num_items]
            targets: Target item indices [batch_size]
            k: Top-K items to consider
        
        Returns:
            Precision@K as float
        """
        _, top_k_items = torch.topk(predictions, k, dim=1)
        targets = targets.unsqueeze(1)
        matches = (top_k_items == targets).float().sum(dim=1)
        precision = matches / k
        return precision.mean().item()
    
    
    @staticmethod
    def mrr(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Mean Reciprocal Rank: Average of 1/rank for target items.
        
        Args:
            predictions: Predicted Q-values [batch_size, num_items]
            targets: Target item indices [batch_size]
        
        Returns:
            MRR as float
        """
        batch_size = predictions.size(0)
        num_items = predictions.size(1)
        
        # Get ranking of all items
        _, rankings = torch.sort(predictions, dim=1, descending=True)
        
        reciprocal_ranks = []
        for i in range(batch_size):
            target_idx = targets[i].item()
            # Find rank of target (1-indexed)
            rank = (rankings[i] == target_idx).nonzero(as_tuple=True)[0].item() + 1
            reciprocal_ranks.append(1.0 / rank)
        
        return np.mean(reciprocal_ranks)
    
    
    @staticmethod
    def q_value_mse(q_pred: torch.Tensor, rewards: torch.Tensor) -> float:
        """
        Mean Squared Error between predicted Q-values and actual rewards.
        
        Args:
            q_pred: Predicted Q-values [batch_size]
            rewards: Actual rewards [batch_size]
        
        Returns:
            MSE as float
        """
        return mean_squared_error(
            rewards.cpu().numpy(),
            q_pred.detach().cpu().numpy()
        )
    
    
    @staticmethod
    def q_value_mae(q_pred: torch.Tensor, rewards: torch.Tensor) -> float:
        """
        Mean Absolute Error between predicted Q-values and actual rewards.
        
        Args:
            q_pred: Predicted Q-values [batch_size]
            rewards: Actual rewards [batch_size]
        
        Returns:
            MAE as float
        """
        return mean_absolute_error(
            rewards.cpu().numpy(),
            q_pred.detach().cpu().numpy()
        )


class UnlearningMetrics:
    """
    Metrics specific to evaluating unlearning effectiveness.
    """
    
    @staticmethod
    def forget_quality(
        forget_performance_after: float,
        forget_performance_before: float
    ) -> float:
        """
        Forget Quality: How much performance degraded on forget set.
        Higher is better (more forgetting).
        
        FQ = (P_before - P_after) / P_before
        
        Returns:
            Forget quality in [0, 1], where 1 = complete forgetting
        """
        if forget_performance_before == 0:
            return 0.0
        return (forget_performance_before - forget_performance_after) / forget_performance_before
    
    
    @staticmethod
    def model_utility(
        retain_performance_after: float,
        retain_performance_baseline: float
    ) -> float:
        """
        Model Utility: How well the model performs on retain set after unlearning.
        Higher is better (utility preserved).
        
        MU = P_after / P_baseline
        
        Returns:
            Model utility, where 1.0 = no degradation
        """
        if retain_performance_baseline == 0:
            return 0.0
        return retain_performance_after / retain_performance_baseline
    
    
    @staticmethod
    def unlearning_efficacy(forget_quality: float, model_utility: float) -> float:
        """
        Overall unlearning efficacy: balance between forgetting and utility.
        
        UE = 2 * (FQ * MU) / (FQ + MU)  (harmonic mean)
        
        Returns:
            Unlearning efficacy in [0, 1]
        """
        if forget_quality + model_utility == 0:
            return 0.0
        return 2 * (forget_quality * model_utility) / (forget_quality + model_utility)
