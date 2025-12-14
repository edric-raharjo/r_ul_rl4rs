import numpy as np
import torch
from torch.utils.data import DataLoader


def _dcg_at_k(rels_sorted, k: int):
    """
    rels_sorted: [B,K] relevance sudah dalam urutan ranking (rank 1..K).
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
    rank_idx: [B,K] index urutan kandidat (descending score)
    """
    # relevance sesuai urutan ranking model
    rel_sorted = torch.gather(labels, dim=1, index=rank_idx)  # [B,K]

    dcg = _dcg_at_k(rel_sorted, k)  # [B]

    # ideal ranking: sort labels descending (positives di atas)
    ideal_idx = torch.argsort(labels, dim=1, descending=True)
    ideal_sorted = torch.gather(labels, dim=1, index=ideal_idx)
    idcg = _dcg_at_k(ideal_sorted, k)  # [B]

    # jika tidak ada positive sama sekali -> idcg=0; biasanya dikeluarkan dari rata-rata
    valid = idcg > 0
    ndcg = torch.zeros_like(dcg)
    ndcg[valid] = dcg[valid] / idcg[valid]
    return ndcg, valid


@torch.no_grad()
def evaluate_hit_ndcg_at_k(model, dataset, ks=(1, 3, 5, 9), device="cuda", batch_size=512):
    """
    Mengembalikan dict metrik: hit@k dan ndcg@k.
    - hit@k: 1 jika ada label=1 di top-k, else 0 (dirata-rata).
    - ndcg@k: normalized DCG (dirata-rata), hanya untuk sample yang punya >=1 positive.
    """
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    ks = sorted(list(ks))
    hits_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    n_total = 0
    n_valid_ndcg = {k: 0 for k in ks}

    for batch in loader:
        state = batch["state"].to(device)
        cand_vecs = batch["next_candidate_item_vecs"].to(device).float()     # [B,K,D]
        cand_mask = batch["next_candidate_mask"].to(device).bool()           # [B,K]
        labels = batch["next_candidate_labels"].to(device).float()           # [B,K]

        q = model.q_values_for_candidates(state, cand_vecs, cand_mask)       # [B,K]

        # Pastikan item masked tidak kepilih top-k
        q = q.masked_fill(~cand_mask, float("-inf"))

        rank_idx = torch.argsort(q, dim=1, descending=True)                  # [B,K]

        B, K = labels.shape
        n_total += B

        # Hit@K
        labels_ranked = torch.gather(labels, 1, rank_idx)                    # [B,K]
        for k in ks:
            k2 = min(k, K)
            hit = (labels_ranked[:, :k2].sum(dim=1) > 0).float()            # [B]
            hits_sum[k] += hit.sum().item()

        # NDCG@K (hanya valid kalau ada positive)
        for k in ks:
            ndcg, valid = _ndcg_at_k_from_labels(labels, rank_idx, k)
            ndcg_sum[k] += ndcg[valid].sum().item()
            n_valid_ndcg[k] += int(valid.sum().item())

    out = {"num_steps": float(n_total)}
    for k in ks:
        out[f"hit@{k}"] = hits_sum[k] / max(n_total, 1)
        out[f"ndcg@{k}"] = ndcg_sum[k] / max(n_valid_ndcg[k], 1)
        out[f"ndcg@{k}_num_valid"] = float(n_valid_ndcg[k])  # debug: berapa step yang punya positive
    return out

def evaluate(model, dataset, device="cuda", ks=(1, 3, 5, 9), batch_size=512):
    return evaluate_hit_ndcg_at_k(model, dataset, ks=ks, device=device, batch_size=batch_size)