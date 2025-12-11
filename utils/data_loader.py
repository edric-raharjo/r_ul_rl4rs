import ast
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


# ---------- Config dataclasses ----------

@dataclass
class LogConfig:
    path: str
    max_click_history: int = 50   # bisa kamu ubah
    slate_size: int = 9           # 9 exposed_items & labels


@dataclass
class ItemConfig:
    path: str
    item_vec_dim: int = 4         # contoh dari screenshot "2,2,1,0.015..."
                                  # nanti bisa kamu set otomatis


# ---------- Utility parsing functions ----------

def parse_int_seq(s: str, sep: str = ",") -> List[int]:
    """Parse string '1,2,3' -> [1,2,3]."""
    s = s.strip().strip('"').strip("'")
    if not s:
        return []
    return [int(x) for x in s.split(sep) if x != ""]


def parse_float_seq(s: str, sep: str = ",") -> List[float]:
    """Parse string '0.1,0.2' -> [0.1,0.2]."""
    s = s.strip().strip('"').strip("'")
    if not s:
        return []
    return [float(x) for x in s.split(sep) if x != ""]


# ---------- Load item table ----------

def load_item_table(cfg: ItemConfig) -> pd.DataFrame:
    """
    Load df_item: columns = [item_id, item_vec, price, location].
    - item_vec (str) -> np.ndarray[dim]
    """
    df_item = pd.read_csv(cfg.path, sep=" ")

    # parse item_vec menjadi array float
    def _parse_vec(s: str) -> np.ndarray:
        return np.asarray(parse_float_seq(s), dtype=np.float32)

    df_item["item_vec_array"] = df_item["item_vec"].apply(_parse_vec)

    # opsional: deteksi dimensi otomatis
    if "item_vec_dim" in cfg.__dict__ and cfg.item_vec_dim is None:
        cfg.item_vec_dim = int(df_item["item_vec_array"].iloc[0].shape[0])

    return df_item

# ---------- Load RL4RS log table ----------

def load_log_table(cfg: LogConfig) -> pd.DataFrame:
    """
    Load df_log: columns ≈
      - user_id
      - user_click_history (str "itemId:timestamp,...")
      - user_protrait (ignored if you don't use)
      - exposed_items (str "id1,id2,...,id9")
      - labels (str "1,1,1,1,1,1,1,1,0")
      - time (timestamp)
    """
    df_log = pd.read_csv(cfg.path, sep=" ")

    # parse click history: "itemId:timestamp,..." -> list[item_id]
    def _parse_click_history(s: str) -> List[int]:
        s = s.strip().strip('"').strip("'")
        if not s:
            return []
        pairs = [p for p in s.split(",") if p != ""]
        item_ids = []
        for p in pairs:
            try:
                item_str, _ = p.split(":")
                item_ids.append(int(item_str))
            except ValueError:
                # kalau formatnya beda, bisa kamu adjust
                continue
        # batasi panjang history
        if len(item_ids) > cfg.max_click_history:
            item_ids = item_ids[-cfg.max_click_history:]
        return item_ids

    # parse exposed_items dan labels
    df_log["click_history_ids"] = df_log["user_click_history"].apply(_parse_click_history)
    df_log["exposed_item_ids"] = df_log["exposed_items"].apply(parse_int_seq)
    df_log["label_list"]        = df_log["labels"].apply(parse_int_seq)

    # sanity check panjang slate
    mask_bad = df_log["exposed_item_ids"].apply(len) != cfg.slate_size
    if mask_bad.any():
        print(f"[WARN] {mask_bad.sum()} rows have exposed_items length != {cfg.slate_size}")
    mask_bad = df_log["label_list"].apply(len) != cfg.slate_size
    if mask_bad.any():
        print(f"[WARN] {mask_bad.sum()} rows have labels length != {cfg.slate_size}")

    return df_log
