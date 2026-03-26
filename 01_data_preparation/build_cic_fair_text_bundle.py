from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robustness_utils import ensure_dir, rp


BIN_LABELS = ["low", "medium", "high", "very_high"]
EXPERIMENTS = [
    "E1_clean",
    "E3_mask_30",
    "E3_mask_50",
    "E3_block_30",
    "E3_block_50",
    "E4_noise_30",
    "E4_noise_50",
]
META_COLS = ["dataset_name", "source_row_id", "source_file", "source_label", "label_name", "label", "text"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a fairness-aligned CIC text bundle from the structured CIC split.")
    ap.add_argument("--input-dir", default="./data/cic_binary_structured_review")
    ap.add_argument("--input-prefix", default="cic_binary_structured_review")
    ap.add_argument("--tier", choices=["mini", "medium", "full"], default="medium")
    ap.add_argument("--output-dir", default="./data/cic_fair_text_csv")
    ap.add_argument("--output-prefix", default="cic_structured_text")
    return ap.parse_args()


def normalize_text_value(value) -> str:
    if value is None:
        return "missing"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "missing"
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.6g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return "missing"
    return s


def fit_bins(train_series: pd.Series, max_bins: int = 4):
    s = pd.to_numeric(train_series, errors="coerce")
    pos = s[(s > 0) & (~s.isna())]
    if pos.empty:
        return None, None
    q = min(max_bins, len(BIN_LABELS))
    edges = np.quantile(pos.to_numpy(), np.linspace(0, 1, q + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.array([0.0, np.inf], dtype=np.float64), ["positive"]
    edges[0] = 0.0
    edges[-1] = np.inf
    return edges.astype(np.float64), BIN_LABELS[: len(edges) - 1]


def apply_bins(series: pd.Series, bins, labels):
    s = pd.to_numeric(series, errors="coerce")
    is_missing = s.isna() | (s < 0)
    is_zero = (s == 0) & (~is_missing)
    if bins is None or labels is None:
        out = pd.Series(["zero"] * len(s), index=series.index, dtype="string")
        out[is_missing] = "missing"
        return out
    s_pos = s.where(s > 0, np.nan)
    cat = pd.cut(s_pos, bins=bins, labels=labels, include_lowest=True)
    out = cat.astype("string").fillna("missing")
    out[is_zero] = "zero"
    out[is_missing] = "missing"
    return out


def build_text(df: pd.DataFrame, num_cols: List[str]) -> pd.Series:
    text_cols = [f"{col}_cat" for col in num_cols]
    return df[text_cols].apply(
        lambda row: " ; ".join(f"{col}={normalize_text_value(row[col])}" for col in text_cols),
        axis=1,
    )


def main() -> None:
    args = parse_args()
    input_dir = rp(args.input_dir)
    output_dir = rp(args.output_dir)
    ensure_dir(output_dir)

    train_clean = os.path.join(input_dir, f"{args.input_prefix}_train_{args.tier}_E1_clean.csv")
    if not os.path.exists(train_clean):
        raise FileNotFoundError(train_clean)
    train_df = pd.read_csv(train_clean)
    num_cols = [c for c in train_df.columns if c not in META_COLS]
    bins_map = {}
    for col in num_cols:
        bins, labels = fit_bins(train_df[col])
        bins_map[col] = {"bins": bins, "labels": labels}

    manifest = {
        "dataset": "CICIDS2017",
        "tier": args.tier,
        "input_prefix": args.input_prefix,
        "output_prefix": args.output_prefix,
        "num_cols": num_cols,
        "cat_cols": [],
        "experiments": EXPERIMENTS,
        "train_rows": int(len(train_df)),
        "val_rows": {},
        "text_numeric_bins": {
            col: {
                "bins": None if meta["bins"] is None else meta["bins"].tolist(),
                "labels": None if meta["labels"] is None else list(meta["labels"]),
            }
            for col, meta in bins_map.items()
        },
    }

    for split, experiments in [("train", ["E1_clean"]), ("val", EXPERIMENTS)]:
        for exp in experiments:
            src = os.path.join(input_dir, f"{args.input_prefix}_{split}_{args.tier}_{exp}.csv")
            if not os.path.exists(src):
                raise FileNotFoundError(src)
            df = pd.read_csv(src)
            out = df.copy()
            for col in num_cols:
                meta = bins_map[col]
                out[f"{col}_cat"] = apply_bins(out[col], meta["bins"], meta["labels"])
            out["text"] = build_text(out, num_cols)
            out_path = os.path.join(output_dir, f"{args.output_prefix}_{split}_{args.tier}_{exp}.csv")
            out.to_csv(out_path, index=False)
            if split == "val":
                manifest["val_rows"][exp] = int(len(out))

    manifest_path = os.path.join(output_dir, f"{args.output_prefix}_{args.tier}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[FAIR CIC TEXT] saved -> {output_dir}")


if __name__ == "__main__":
    main()
