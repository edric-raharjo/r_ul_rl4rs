# dataset/rl4rs_dataset_9step_candidates.py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple


class RL4RSDataset9Step(Dataset):
    """
    1 baris df_log (1 episode dengan 9 slot) -> 9 transition.

    Candidate action set untuk next_state:
      - memakai slate 9 item yang sama (exposed_item_ids) sebagai kandidat (K=9)
      - mask dapat menolak kandidat yang sudah ada di history / sudah dipakai sebelumnya

    Output per transition:
      - state
      - item_vec          (aksi/ item pada slot k dari log)
      - reward            (tier-weighted + gating 3-3-3)
      - done
      - next_state
      - next_candidate_item_vecs  (K=9)
      - next_candidate_mask       (K=9)
    """

    def __init__(self,
                 df_log,
                 df_item,
                 slate_size: int = 9,
                 use_tier_flags: bool = True,
                 tier_weights: Tuple[float, float, float] = (1.0, 2.0, 4.0),
                 exclude_history_candidates: bool = True,
                 exclude_used_candidates: bool = True):
        super().__init__()
        self.df_item = df_item.set_index("item_id")
        self.item_vec_dim = self.df_item["item_vec_array"].iloc[0].shape[0]

        self.slate_size = slate_size
        self.use_tier_flags = use_tier_flags
        self.w1, self.w2, self.w3 = tier_weights

        self.exclude_history_candidates = exclude_history_candidates
        self.exclude_used_candidates = exclude_used_candidates

        self.transitions: List[Dict[str, Any]] = []
        self._build_transitions(df_log)

    # ---------- state building ----------

    def _history_embedding(self, history_ids: List[int]) -> np.ndarray:
        vecs = []
        for iid in history_ids:
            if iid in self.df_item.index:
                vecs.append(self.df_item.loc[iid, "item_vec_array"])
        if len(vecs) == 0:
            return np.zeros(self.item_vec_dim, dtype=np.float32)
        return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)

    def _tier_flags(self, bought_tier1: int, bought_tier2: int, bought_tier3: int) -> np.ndarray:
        return np.array(
            [bought_tier1 / 3.0, bought_tier2 / 3.0, bought_tier3 / 3.0],
            dtype=np.float32
        )

    # ---------- reward (tier-weighted + gating 3-3-3) ----------

    def _compute_rewards_3_3_3_weighted(self, labels: List[int]) -> List[float]:
        """
        Reward per slot:
        - Tier1 slot 1-3: label==1 -> w1
        - Tier2 slot 4-6 only if first 3 labels are 1 -> label==1 -> w2
        - Tier3 slot 7-9 only if first 6 labels are 1 -> label==1 -> w3
        """
        r = [0.0] * self.slate_size

        # tier1 always available
        for k in range(0, 3):
            if labels[k] == 1:
                r[k] = self.w1

        tier1_full = all(labels[i] == 1 for i in range(0, 3))

        # tier2 only if tier1 full
        if tier1_full:
            for k in range(3, 6):
                if labels[k] == 1:
                    r[k] = self.w2

        tier2_full = tier1_full and all(labels[i] == 1 for i in range(3, 6))

        # tier3 only if tier2 full
        if tier2_full:
            for k in range(6, 9):
                if labels[k] == 1:
                    r[k] = self.w3

        return r

    # ---------- candidates (K=9 slate) ----------

    def _slate_candidate_vecs(self, exposed_ids: List[int]) -> np.ndarray:
        """
        Convert exposed item_ids (len=9) -> [9, item_dim] vectors.
        Unknown item -> zeros.
        """
        vecs = []
        for iid in exposed_ids:
            if iid in self.df_item.index:
                vecs.append(self.df_item.loc[iid, "item_vec_array"])
            else:
                vecs.append(np.zeros(self.item_vec_dim, dtype=np.float32))
        return np.stack(vecs, axis=0).astype(np.float32)

    def _build_candidate_mask(self,
                              exposed_ids: List[int],
                              history_ids: List[int],
                              used_ids: List[int]) -> np.ndarray:
        """
        Mask untuk 9 kandidat slate.
        - exclude_history_candidates: kandidat yang sudah ada di history -> False
        - exclude_used_candidates: kandidat yang sudah dipakai di slot sebelumnya -> False
        """
        mask = np.ones(self.slate_size, dtype=bool)

        if self.exclude_history_candidates:
            hist_set = set(history_ids)
            for j, iid in enumerate(exposed_ids):
                if iid in hist_set:
                    mask[j] = False

        if self.exclude_used_candidates:
            used_set = set(used_ids)
            for j, iid in enumerate(exposed_ids):
                if iid in used_set:
                    mask[j] = False

        # fallback: kalau semuanya False, biarkan semua True supaya tidak -inf semua
        if mask.sum() == 0:
            mask[:] = True

        return mask

    # ---------- build transitions ----------

    def _build_transitions(self, df_log):
        for row in df_log.itertuples(index=False):
            click_history: List[int] = getattr(row, "click_history_ids")
            exposed_ids: List[int] = getattr(row, "exposed_item_ids")
            labels: List[int] = getattr(row, "label_list")

            if len(exposed_ids) != self.slate_size or len(labels) != self.slate_size:
                continue

            rewards = self._compute_rewards_3_3_3_weighted(labels)
            slate_vecs = self._slate_candidate_vecs(exposed_ids)  # [9, D]

            history_ids = list(click_history)
            used_ids: List[int] = []

            bought_tier1 = 0
            bought_tier2 = 0
            bought_tier3 = 0

            for k in range(self.slate_size):
                slot_idx = k + 1
                item_id = exposed_ids[k]
                reward_k = float(rewards[k])

                # ----- state -----
                hist_emb = self._history_embedding(history_ids)
                if self.use_tier_flags:
                    state_vec = np.concatenate(
                        [hist_emb, self._tier_flags(bought_tier1, bought_tier2, bought_tier3)],
                        axis=0
                    )
                else:
                    state_vec = hist_emb

                # ----- action item_vec (item pada slot k dari log) -----
                if item_id in self.df_item.index:
                    item_vec = self.df_item.loc[item_id, "item_vec_array"]
                else:
                    item_vec = np.zeros(self.item_vec_dim, dtype=np.float32)

                # ----- update history & used -----
                history_ids.append(item_id)
                used_ids.append(item_id)

                # ----- update tier counters: gunakan valid_purchase -----
                # reward_k > 0 berarti label==1 dan tier unlocked
                if reward_k > 0.0:
                    if slot_idx <= 3:
                        bought_tier1 += 1
                    elif slot_idx <= 6:
                        bought_tier2 += 1
                    else:
                        bought_tier3 += 1

                # ----- next_state -----
                next_hist_emb = self._history_embedding(history_ids)
                if self.use_tier_flags:
                    next_state_vec = np.concatenate(
                        [next_hist_emb, self._tier_flags(bought_tier1, bought_tier2, bought_tier3)],
                        axis=0
                    )
                else:
                    next_state_vec = next_hist_emb

                done = 1.0 if slot_idx == self.slate_size else 0.0

                # ----- next candidates & mask -----
                next_candidate_item_vecs = slate_vecs.copy()  # [9, D]
                next_candidate_mask = self._build_candidate_mask(exposed_ids, history_ids, used_ids)  # [9]
                next_candidate_labels = np.array(labels, dtype=np.float32)  # [9]

                # ----- store transition -----
                self.transitions.append({
                    "state": torch.from_numpy(state_vec).float(),
                    "item_vec": torch.from_numpy(item_vec).float(),
                    "reward": torch.tensor(reward_k, dtype=torch.float32),
                    "done": torch.tensor(done, dtype=torch.float32),
                    "next_state": torch.from_numpy(next_state_vec).float(),
                    "next_candidate_item_vecs": torch.from_numpy(next_candidate_item_vecs).float(),
                    "next_candidate_mask": torch.from_numpy(next_candidate_mask),
                    "next_candidate_labels": torch.from_numpy(next_candidate_labels).float(),
                })

    # ---------- Dataset API ----------

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.transitions[idx]
