"""
prepare_a1_unsw15.py

End-to-end data preparation for Scheme A1 (frozen encoder + linear probe).

What this script does (paper-grade protocol):
1) Load RAW UNSW-NB15 CSV.
2) Clean dtypes and missing values WITHOUT leaking validation info.
3) Map attack_cat -> integer label using config.LABEL_MAP.
4) Stratified train/val split.
5) Fit numeric bin edges on TRAIN ONLY, then apply to both train/val.
6) Build compact, content-only embedding text (no instruction prefix).
7) Save full/medium/mini tiers:
   - PROMPT_TRAIN_CSV / PROMPT_VAL_CSV          (full)
   - PROMPT_TRAIN_CSV_MEDIUM / PROMPT_VAL_CSV_MEDIUM
   - PROMPT_TRAIN_CSV_MINI / PROMPT_VAL_CSV_MINI

NOTE:
- We keep the same output variable names (PROMPT_*_CSV) to minimize changes
  to your existing probe script. The output column used for embedding is `text`.
"""

from __future__ import annotations

import os
import re
import sys
import json
import csv
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


SUPPORTED_EXPERIMENTS = [
    "E1_clean",
    "E3_mask_10",
    "E3_mask_30",
    "E3_mask_50",
    "E3_block_10",
    "E3_block_30",
    "E3_block_50",
    "E4_noise_10",
    "E4_noise_30",
    "E4_noise_50",
]


# --- Path config (compatible with your project layout) ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import (
    RAW_DATA_CSV,
    FILTERED_DATA_CSV,
    PROMPT_TRAIN_CSV,
    PROMPT_VAL_CSV,
    PROMPT_TRAIN_CSV_MINI,
    PROMPT_VAL_CSV_MINI,
    PROMPT_TRAIN_CSV_MEDIUM,
    PROMPT_VAL_CSV_MEDIUM,
    LABEL_MAP,
)


def apply_missing_mask(df: pd.DataFrame, cols: List[str], ratio: float, seed: int = 42):
    """
    Randomly mask numeric columns with NaN (val only).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    for c in cols:
        mask = rng.random(len(df)) < ratio
        df.loc[mask, c] = np.nan
    return df


def apply_noise(df: pd.DataFrame, cols: List[str], ratio: float, seed: int = 42):
    """
    Apply multiplicative noise to numeric columns (val only).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    for c in cols:
        values = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float, copy=True)
        mask = np.isfinite(values) & (values > 0)
        values[mask] = values[mask] * (1.0 + rng.uniform(-ratio, ratio, size=mask.sum()))
        df[c] = values
    return df


def apply_block_missing_mask(
    df: pd.DataFrame,
    blocks: List[List[str]],
    ratio: float,
    seed: int = 42,
):
    """
    Randomly mask predefined feature groups together.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    for block in blocks:
        present = [c for c in block if c in out.columns]
        if not present:
            continue
        mask = rng.random(len(out)) < ratio
        out.loc[mask, present] = np.nan
    return out


def resolve_path(path_str: str) -> str:
    if isinstance(path_str, str) and (
        path_str.startswith("./") or path_str.startswith("../")
    ):
        return os.path.join(project_root, path_str)
    return path_str


def with_exp(path: str, experiment: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}_{experiment}{ext}"


def override_parent_dir(path: str, new_dir: str) -> str:
    if not new_dir:
        return path
    return os.path.join(resolve_path(new_dir), os.path.basename(path))


RAW_DATA_CSV = resolve_path(RAW_DATA_CSV)
FILTERED_DATA_CSV = resolve_path(FILTERED_DATA_CSV)
PROMPT_TRAIN_CSV = resolve_path(PROMPT_TRAIN_CSV)
PROMPT_VAL_CSV = resolve_path(PROMPT_VAL_CSV)

PROMPT_TRAIN_CSV_MINI = resolve_path(PROMPT_TRAIN_CSV_MINI)
PROMPT_VAL_CSV_MINI = resolve_path(PROMPT_VAL_CSV_MINI)

PROMPT_TRAIN_CSV_MEDIUM = resolve_path(PROMPT_TRAIN_CSV_MEDIUM)
PROMPT_VAL_CSV_MEDIUM = resolve_path(PROMPT_VAL_CSV_MEDIUM)
# ---------------------------  Config  ---------------------------

# Columns used for A1
A1_BASE_COLS = [
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "sload",
    "dload",
    "tcprtt",
    "ct_srv_src",
    "attack_cat",
]

# Numeric columns to bin into semantic categories
A1_BIN_COLS = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "sload",
    "dload",
    "tcprtt",
    "ct_srv_src",
]

A1_CAT_COLS = ["proto", "service", "state"]

BIN_LABELS = ["low", "medium", "high", "very_high"]
BIN_TARGET_VOCAB = ["missing", "zero", "positive", "low", "medium", "high", "very_high"]
BLOCK_GROUPS = [
    ["dur", "spkts", "dpkts", "sbytes", "dbytes"],
    ["sload", "dload", "tcprtt", "ct_srv_src"],
]

_TRAILING_PUNCT_RE = re.compile(r"[\.。．,，;；:：\s]+$")


def norm_str(x, *, default: str = "UNK", dash_to: str = "NONE") -> str:
    """Normalize categorical fields, avoiding 'nan' string pitfalls."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return default
    if s == "-":
        return dash_to
    return s


def norm_attack_cat(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    s = _TRAILING_PUNCT_RE.sub("", s)  # remove trailing punctuation / spaces
    return s


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def source_data_csv() -> str:
    if os.path.exists(RAW_DATA_CSV):
        return RAW_DATA_CSV
    if os.path.exists(FILTERED_DATA_CSV):
        return FILTERED_DATA_CSV
    raise FileNotFoundError(
        f"Neither RAW_DATA_CSV nor FILTERED_DATA_CSV exists: {RAW_DATA_CSV} | {FILTERED_DATA_CSV}"
    )


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Return a stratified subsample (or full df if n>=len). Falls back to random sample if needed."""
    if n <= 0 or n >= len(df):
        return df.copy()
    y = df["label"].to_numpy()
    vc = pd.Series(y).value_counts()
    if vc.min() < 2:
        return df.sample(n=n, random_state=seed).copy()
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(sss.split(np.zeros(len(df)), y))
    return df.iloc[idx].copy()


def fit_bins(
    train_series: pd.Series, max_bins: int = 4
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Fit bin edges on TRAIN ONLY.
    - Uses quantiles over positive values.
    - Ensures coverage by setting last edge to +inf.
    Returns (bins, labels). If all zeros -> (None, None).
    """
    s = pd.to_numeric(train_series, errors="coerce")
    pos = s[(s > 0) & (~s.isna())]
    if pos.empty:
        return None, None

    q = min(max_bins, len(BIN_LABELS))
    edges = np.quantile(pos.to_numpy(), np.linspace(0, 1, q + 1))
    edges = np.unique(edges)

    if len(edges) <= 2:
        bins = np.array([0.0, np.inf], dtype=np.float64)
        labels = ["positive"]
        return bins, labels

    edges[0] = 0.0
    edges[-1] = np.inf
    bins = edges.astype(np.float64)
    labels = BIN_LABELS[: len(bins) - 1]
    return bins, labels


def apply_bins(
    series: pd.Series,
    bins: Optional[np.ndarray],
    labels: Optional[List[str]],
    *,
    zero_label: str = "zero",
    missing_label: str = "missing",
) -> pd.Series:
    """
    Apply pre-fitted bins to a series.
    - NaN or negative -> missing_label
    - exact 0 -> zero_label
    - >0 -> one of labels via pd.cut
    """
    s = pd.to_numeric(series, errors="coerce")
    is_missing = s.isna() | (s < 0)
    is_zero = (s == 0) & (~is_missing)

    if bins is None or labels is None:
        out = pd.Series([zero_label] * len(s), index=series.index, dtype="string")
        out[is_missing] = missing_label
        return out

    s_pos = s.where(s > 0, np.nan)
    cat = pd.cut(s_pos, bins=bins, labels=labels, include_lowest=True)
    out = cat.astype("string")
    out = out.fillna(missing_label)  # safety
    out[is_zero] = zero_label
    out[is_missing] = missing_label
    return out


def build_kv_text(row: pd.Series) -> str:
    """Content-only embedding text for A1: short, stable, and auditable."""
    parts = [
        f"proto={row['proto']}",
        f"state={row['state']}",
        f"service={row['service']}",
    ]
    for col in A1_BIN_COLS:
        parts.append(f"{col}={row.get(col + '_cat', 'missing')}")
    return " ; ".join(parts)


def build_nl_text(row: pd.Series) -> str:
    """Optional natural-language description (no instruction prefix)."""
    proto = row["proto"]
    state = row["state"]
    service = row["service"]

    dur_cat = row.get("dur_cat", "missing")
    s_pkts = row.get("spkts_cat", "missing")
    d_pkts = row.get("dpkts_cat", "missing")
    s_bytes = row.get("sbytes_cat", "missing")
    d_bytes = row.get("dbytes_cat", "missing")
    sload = row.get("sload_cat", "missing")
    dload = row.get("dload_cat", "missing")
    tcprtt = row.get("tcprtt_cat", "missing")
    ct = row.get("ct_srv_src_cat", "missing")

    return (
        f"A {proto} flow in {state} state using service {service}. "
        f"Duration={dur_cat}. "
        f"Source packets={s_pkts}, bytes={s_bytes}; destination packets={d_pkts}, bytes={d_bytes}. "
        f"Source load={sload}, destination load={dload}. "
        f"TCP RTT={tcprtt}. Prior same-service count={ct}."
    )


def add_observation_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in A1_CAT_COLS:
        values = out[col].astype("string")
        out[f"obs_{col}"] = (~values.isna() & (values != "UNK")).astype(np.int64)
    for col in A1_BIN_COLS:
        values = pd.to_numeric(out[col], errors="coerce")
        out[f"obs_{col}"] = (~values.isna() & (values >= 0)).astype(np.int64)
    obs_num_cols = [f"obs_{col}" for col in A1_BIN_COLS if f"obs_{col}" in out.columns]
    out["num_missing_ratio"] = 1.0 - out[obs_num_cols].mean(axis=1)
    out["num_observed_ratio"] = out[obs_num_cols].mean(axis=1)
    return out


def build_output_columns(include_text_nl: bool) -> List[str]:
    cols = [
        "source_row_id",
        "label",
        "attack_cat",
        "text",
        "num_missing_ratio",
        "num_observed_ratio",
    ]
    if include_text_nl:
        cols.append("text_nl")
    cols.extend(A1_CAT_COLS)
    cols.extend(A1_BIN_COLS)
    cols.extend([f"{col}_cat" for col in A1_BIN_COLS])
    cols.extend([f"obs_{col}" for col in A1_CAT_COLS + A1_BIN_COLS])
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment",
        choices=SUPPORTED_EXPERIMENTS,
        default="E1_clean",
        help="Validation perturbation setting encoded into output filenames.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument(
        "--write_filtered",
        action="store_true",
        help="Also write FILTERED_DATA_CSV for inspection.",
    )
    ap.add_argument("--mini_train", type=int, default=20000)
    ap.add_argument("--mini_val", type=int, default=5000)
    ap.add_argument("--medium_train", type=int, default=100000)
    ap.add_argument("--medium_val", type=int, default=20000)
    ap.add_argument(
        "--add_nl_text",
        action="store_true",
        help="Also save text_nl column (ablation).",
    )
    ap.add_argument(
        "--split_index",
        type=str,
        default="./data/prompt_csv/splits_unsw15_seed{seed}.npz",
        help="Cache train/val indices to keep splits fixed across experiments.",
    )
    ap.add_argument(
        "--prompt_dir_override",
        type=str,
        default="",
        help="Optional output directory for prompt CSVs to avoid overwriting split-specific runs.",
    )
    args = ap.parse_args()

    global PROMPT_TRAIN_CSV, PROMPT_VAL_CSV
    global PROMPT_TRAIN_CSV_MINI, PROMPT_VAL_CSV_MINI
    global PROMPT_TRAIN_CSV_MEDIUM, PROMPT_VAL_CSV_MEDIUM
    prompt_train_csv = with_exp(PROMPT_TRAIN_CSV, args.experiment)
    prompt_val_csv = with_exp(PROMPT_VAL_CSV, args.experiment)
    prompt_train_csv_mini = with_exp(PROMPT_TRAIN_CSV_MINI, args.experiment)
    prompt_val_csv_mini = with_exp(PROMPT_VAL_CSV_MINI, args.experiment)
    prompt_train_csv_medium = with_exp(PROMPT_TRAIN_CSV_MEDIUM, args.experiment)
    prompt_val_csv_medium = with_exp(PROMPT_VAL_CSV_MEDIUM, args.experiment)
    if args.prompt_dir_override:
        prompt_train_csv = override_parent_dir(prompt_train_csv, args.prompt_dir_override)
        prompt_val_csv = override_parent_dir(prompt_val_csv, args.prompt_dir_override)
        prompt_train_csv_mini = override_parent_dir(prompt_train_csv_mini, args.prompt_dir_override)
        prompt_val_csv_mini = override_parent_dir(prompt_val_csv_mini, args.prompt_dir_override)
        prompt_train_csv_medium = override_parent_dir(
            prompt_train_csv_medium, args.prompt_dir_override
        )
        prompt_val_csv_medium = override_parent_dir(
            prompt_val_csv_medium, args.prompt_dir_override
        )

    split_path = args.split_index.format(seed=args.seed) if "{seed}" in args.split_index else args.split_index

    src_csv = source_data_csv()
    print(f"[LOAD] {src_csv}")
    df = pd.read_csv(src_csv, low_memory=False)
    df["source_row_id"] = np.arange(len(df), dtype=np.int64)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    existing = ["source_row_id"] + [c for c in A1_BASE_COLS if c in df.columns]
    missing_cols = set(A1_BASE_COLS) - set(existing)
    if missing_cols:
        print(f"[WARN] Missing columns in raw CSV (ignored): {sorted(missing_cols)}")
    df = df[existing].copy()

    df["attack_cat"] = df["attack_cat"].apply(norm_attack_cat)

    label2id = {v: k for k, v in LABEL_MAP.items()}
    df["label"] = df["attack_cat"].map(label2id)

    bad = df["label"].isna().sum()
    if bad > 0:
        print(f"[WARN] Drop {bad} rows with unmapped attack_cat.")
        df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    for c in A1_CAT_COLS:
        if c in df.columns:
            df[c] = df[c].apply(norm_str).astype("string")

    for c in A1_BIN_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    num_cols = [c for c in A1_BIN_COLS if c in df.columns]
    miss_counts = df[num_cols].isna().sum()
    neg_counts = (df[num_cols] < 0).sum()
    print("[QC] numeric NaN counts:\n", miss_counts.to_string())
    print("[QC] numeric negative counts:\n", neg_counts.to_string())

    if args.write_filtered and src_csv != FILTERED_DATA_CSV:
        ensure_dir(FILTERED_DATA_CSV)
        df.to_csv(FILTERED_DATA_CSV, index=False)
        print(f"[SAVE] filtered -> {FILTERED_DATA_CSV}")

    y = df["label"]
    vc = y.value_counts()
    stratify_arg = y if vc.min() >= 2 else None
    if stratify_arg is None:
        print("[WARN] Some class has <2 samples; stratify disabled.")

    if os.path.exists(split_path):
        idx = np.load(split_path)
        train_idx = idx["train_idx"]
        val_idx = idx["val_idx"]
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        print(f"[SPLIT] loaded cached indices -> {split_path}")
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=args.val_ratio,
            random_state=args.seed,
            stratify=stratify_arg,
        )
        np.savez(split_path, train_idx=train_df.index.to_numpy(), val_idx=val_df.index.to_numpy())
        print(f"[SPLIT] saved indices -> {split_path}")
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"[SPLIT] train={len(train_df)} val={len(val_df)}")

    # ============================================================
    # Apply perturbation on VALIDATION ONLY (E3 / E4)
    # ============================================================
    if args.experiment.startswith("E3_mask"):
        ratio = float(args.experiment.split("_")[-1]) / 100.0
        val_df = apply_missing_mask(val_df, A1_BIN_COLS, ratio)
    elif args.experiment.startswith("E3_block"):
        ratio = float(args.experiment.split("_")[-1]) / 100.0
        val_df = apply_block_missing_mask(val_df, BLOCK_GROUPS, ratio)

    elif args.experiment.startswith("E4_noise"):
        ratio = float(args.experiment.split("_")[-1]) / 100.0
        val_df = apply_noise(val_df, A1_BIN_COLS, ratio)

    # E1_clean: do nothing

    bins_json: Dict[str, Dict] = {}
    for c in A1_BIN_COLS:
        if c not in train_df.columns:
            continue
        bins, labels = fit_bins(train_df[c], max_bins=len(BIN_LABELS))
        train_df[c + "_cat"] = apply_bins(train_df[c], bins, labels)
        val_df[c + "_cat"] = apply_bins(val_df[c], bins, labels)
        bins_json[c] = {
            "bins": None if bins is None else [float(x) for x in bins.tolist()],
            "labels": labels,
        }

    train_df["text"] = train_df.apply(build_kv_text, axis=1)
    val_df["text"] = val_df.apply(build_kv_text, axis=1)

    if args.add_nl_text:
        train_df["text_nl"] = train_df.apply(build_nl_text, axis=1)
        val_df["text_nl"] = val_df.apply(build_nl_text, axis=1)

    train_df = add_observation_columns(train_df)
    val_df = add_observation_columns(val_df)

    out_cols = build_output_columns(args.add_nl_text)

    train_out = train_df[out_cols].copy()
    val_out = val_df[out_cols].copy()

    ensure_dir(prompt_train_csv)
    train_out.to_csv(prompt_train_csv, index=False, quoting=csv.QUOTE_ALL)
    val_out.to_csv(prompt_val_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"[SAVE] full train -> {prompt_train_csv}  (rows={len(train_out)})")
    print(f"[SAVE] full val   -> {prompt_val_csv}    (rows={len(val_out)})")

    bins_path = os.path.join(
        os.path.dirname(prompt_train_csv), f"a1_bins_train_only_{args.experiment}.json"
    )
    ensure_dir(bins_path)
    with open(bins_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "val_ratio": args.val_ratio,
                "bin_labels": BIN_LABELS,
                "bin_target_vocab": BIN_TARGET_VOCAB,
                "bin_cols": A1_BIN_COLS,
                "bins": bins_json,
                "block_groups": BLOCK_GROUPS,
                "label_map": LABEL_MAP,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[SAVE] bins json -> {bins_path}")

    if args.medium_train > 0 or args.medium_val > 0:
        med_tr = stratified_sample(
            train_out, min(args.medium_train, len(train_out)), seed=args.seed
        )
        med_va = stratified_sample(
            val_out, min(args.medium_val, len(val_out)), seed=args.seed + 1
        )
        ensure_dir(prompt_train_csv_medium)
        med_tr.to_csv(prompt_train_csv_medium, index=False, quoting=csv.QUOTE_ALL)
        med_va.to_csv(prompt_val_csv_medium, index=False, quoting=csv.QUOTE_ALL)
        print(f"[SAVE] medium train -> {prompt_train_csv_medium}  (rows={len(med_tr)})")
        print(f"[SAVE] medium val   -> {prompt_val_csv_medium}    (rows={len(med_va)})")

    if args.mini_train > 0 or args.mini_val > 0:
        mini_tr = stratified_sample(
            train_out, min(args.mini_train, len(train_out)), seed=args.seed + 2
        )
        mini_va = stratified_sample(
            val_out, min(args.mini_val, len(val_out)), seed=args.seed + 3
        )
        ensure_dir(prompt_train_csv_mini)
        mini_tr.to_csv(prompt_train_csv_mini, index=False, quoting=csv.QUOTE_ALL)
        mini_va.to_csv(prompt_val_csv_mini, index=False, quoting=csv.QUOTE_ALL)
        print(f"[SAVE] mini train -> {prompt_train_csv_mini}  (rows={len(mini_tr)})")
        print(f"[SAVE] mini val   -> {prompt_val_csv_mini}    (rows={len(mini_va)})")

    print("[DONE] A1 dataset preparation complete.")


if __name__ == "__main__":
    main()
