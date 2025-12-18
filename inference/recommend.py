from typing import List, Dict, Optional
import numpy as np
import torch

# pastikan QNetwork kamu punya method q_values_for_candidates
from scripts.model.dqn_agent import sQNetwork


def build_history_embedding(history_item_ids: List[int],
                            item_vec_lookup: Dict[int, np.ndarray],
                            item_dim: int) -> np.ndarray:
    vecs = []
    for iid in history_item_ids:
        v = item_vec_lookup.get(iid, None)
        if v is not None:
            vecs.append(v)
    if len(vecs) == 0:
        return np.zeros(item_dim, dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)


def tier_from_progress(bought_tier1: int, bought_tier2: int) -> int:
    if bought_tier1 < 3:
        return 1
    if bought_tier2 < 3:
        return 2
    return 3


@torch.no_grad()
def recommend_9_items(model: QNetwork,
                      history_item_ids: List[int],
                      item_vec_lookup: Dict[int, np.ndarray],
                      all_item_ids: List[int],
                      all_item_vecs: np.ndarray,
                      device: str = "cuda",
                      exclude_history: bool = True,
                      candidate_item_ids: Optional[List[int]] = None
                      ) -> List[int]:

    model.eval()
    model.to(device)

    # ----- candidate pool -----
    if candidate_item_ids is None:
        cand_ids = np.array(all_item_ids, dtype=np.int64)
        cand_vecs = all_item_vecs.astype(np.float32)
    else:
        # amanin: filter hanya yang ada di lookup
        filtered = [i for i in candidate_item_ids if i in item_vec_lookup]
        if len(filtered) == 0:
            raise ValueError("candidate_item_ids kosong setelah difilter (tidak ada yang ada di item_vec_lookup).")
        cand_ids = np.array(filtered, dtype=np.int64)
        cand_vecs = np.stack([item_vec_lookup[i] for i in cand_ids], axis=0).astype(np.float32)

    N, item_dim = cand_vecs.shape

    # ----- state awal -----
    history_ids = list(history_item_ids)
    bought_tier1 = 0
    bought_tier2 = 0
    bought_tier3 = 0

    recs: List[int] = []

    for slot_idx in range(1, 10):  # 1..9
        # 1) build state (history embedding + tier flags)
        hist_emb = build_history_embedding(history_ids, item_vec_lookup, item_dim)
        tier_flags = np.array(
            [bought_tier1/3.0, bought_tier2/3.0, bought_tier3/3.0],
            dtype=np.float32
        )
        state_vec = np.concatenate([hist_emb, tier_flags], axis=0).astype(np.float32)
        state = torch.from_numpy(state_vec).float().unsqueeze(0).to(device)  # [1, state_dim]

        # 2) mask kandidat
        mask = np.ones(N, dtype=bool)
        if exclude_history:
            hist_set = set(history_ids)
            mask &= np.array([iid not in hist_set for iid in cand_ids], dtype=bool)

        if mask.sum() == 0:
            mask[:] = True  # fallback

        cand_vecs_t = torch.from_numpy(cand_vecs).float().unsqueeze(0).to(device)      # [1, N, item_dim]
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device).bool()                # [1, N] bool

        # 3) hitung Q semua kandidat, pilih terbaik
        q_all = model.q_values_for_candidates(state, cand_vecs_t, mask_t)              # [1, N]
        best_idx = int(torch.argmax(q_all, dim=1).item())
        best_item_id = int(cand_ids[best_idx])

        recs.append(best_item_id)

        # 4) update history & progress tier (optimistic)
        history_ids.append(best_item_id)
        if slot_idx <= 3:
            bought_tier1 += 1
        elif slot_idx <= 6:
            bought_tier2 += 1
        else:
            bought_tier3 += 1

    return recs

