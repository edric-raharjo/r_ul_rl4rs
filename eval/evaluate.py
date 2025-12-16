import numpy as np
import torch
from torch.utils.data import DataLoader


def _dcg_at_k(rels_sorted, k: int):
    """
    rels_sorted: [B,K] relevance already ordered by rank (rank 1..K).
    DCG@k = sum_{i=1..k} rel_i / log2(i+1)
    """
    device = rels_sorted.device
    k = min(k, rels_sorted.shape[1])
    rels_k = rels_sorted[:, :k]
    discounts = torch.log2(torch.arange(2, k + 2, device=device).float())  # [k], 2..k+1
    return (rels_k / discounts.unsqueeze(0)).sum(dim=1)  # [B]


def _ndcg_at_k_from_labels(labels, rank_idx, k: int):
    """
    labels: [B,K] (0/1)
    rank_idx: [B,K] indices of candidates in descending score order
    """
    # relevance in the model's ranked order
    rel_sorted = torch.gather(labels, dim=1, index=rank_idx)  # [B,K]

    dcg = _dcg_at_k(rel_sorted, k)  # [B]

    # ideal ranking: sort labels descending (positives first)
    ideal_idx = torch.argsort(labels, dim=1, descending=True)
    ideal_sorted = torch.gather(labels, dim=1, index=ideal_idx)
    idcg = _dcg_at_k(ideal_sorted, k)  # [B]

    # if no positives at all -> idcg=0; exclude from mean
    valid = idcg > 0
    ndcg = torch.zeros_like(dcg)
    ndcg[valid] = dcg[valid] / idcg[valid]
    return ndcg, valid


@torch.no_grad()
def evaluate_hit_ndcg_at_k(model, dataset, ks=(1, 3, 5, 9), device="cuda", batch_size=512):
    """
    Returns a dict of metrics: hit@k and ndcg@k.
    - hit@k: 1 if any label=1 in top-k, else 0 (averaged).
    - ndcg@k: normalized DCG (averaged) over samples with >=1 positive.
    """
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    ks = sorted(list(ks))
    hits_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    n_total = 0
    n_valid_ndcg = {k: 0 for k in ks}

    for batch in loader:
        state = batch["state"].to(device)                                   # [B, state_dim]
        cand_ids = batch["next_candidate_ids"].to(device).long()           # [B,K]
        cand_mask = batch["next_candidate_mask"].to(device).bool()         # [B,K]
        labels = batch["next_candidate_labels"].to(device).float()         # [B,K]

        # Q-values for candidates using catalog network
        q = model.q_values_for_candidates(
            state=state,
            candidate_ids=cand_ids,
            candidate_mask=cand_mask,
        )                                                                   # [B,K]

        # Ensure masked items cannot be chosen
        q = q.masked_fill(~cand_mask, float("-inf"))

        # Ranking indices (descending Q)
        rank_idx = torch.argsort(q, dim=1, descending=True)                # [B,K]

        B, K = labels.shape
        n_total += B

        # Hit@K
        labels_ranked = torch.gather(labels, 1, rank_idx)                  # [B,K]
        for k in ks:
            k2 = min(k, K)
            hit = (labels_ranked[:, :k2].sum(dim=1) > 0).float()           # [B]
            hits_sum[k] += hit.sum().item()

        # NDCG@K (only for samples with at least one positive)
        for k in ks:
            ndcg, valid = _ndcg_at_k_from_labels(labels, rank_idx, k)
            ndcg_sum[k] += ndcg[valid].sum().item()
            n_valid_ndcg[k] += int(valid.sum().item())

    out = {"num_steps": float(n_total)}
    for k in ks:
        out[f"hit@{k}"] = hits_sum[k] / max(n_total, 1)
        out[f"ndcg@{k}"] = ndcg_sum[k] / max(n_valid_ndcg[k], 1)
        out[f"ndcg@{k}_num_valid"] = float(n_valid_ndcg[k])
    return out


def evaluate(model, dataset, device="cuda", ks=(1, 3, 5, 9), batch_size=512):
    return evaluate_hit_ndcg_at_k(model, dataset, ks=ks, device=device, batch_size=batch_size)
