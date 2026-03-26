from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import List

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import FILTERED_DATA_CSV, LABEL_MAP
from nn_tabular_baseline import (
    apply_block_missing_mask,
    apply_missing_mask,
    apply_noise,
    infer_block_groups,
    infer_feature_columns,
    parse_csv_list,
    rp,
    stratified_cap,
)
from robustness_utils import DEFAULT_VAL_EXPERIMENTS, ensure_dir


PER_CLASS_CAP = {"mini": 300, "medium": 3000}
BIN_LABELS = ["low", "medium", "high", "very_high"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a fairness-aligned UNSW text bundle from the same curated structured split used by tabular models."
    )
    ap.add_argument("--dataset-csv", default=FILTERED_DATA_CSV)
    ap.add_argument("--tier", choices=["mini", "medium", "full"], default="medium")
    ap.add_argument("--split-seeds", default="42,52,62")
    ap.add_argument("--split-index", default="./data/prompt_csv/splits_unsw15_seed{seed}.npz")
    ap.add_argument("--feature-mode", choices=["available", "legacy_selected"], default="available")
    ap.add_argument("--output-dir", default="./data/fair_text_csv")
    ap.add_argument("--data-prefix", default="unsw_structured_text")
    ap.add_argument("--val-experiments", default=",".join(DEFAULT_VAL_EXPERIMENTS))
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


def build_text(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> pd.Series:
    text_cols = list(cat_cols) + [f"{col}_cat" for col in num_cols]
    return df[text_cols].apply(
        lambda row: " ; ".join(f"{col}={normalize_text_value(row[col])}" for col in text_cols),
        axis=1,
    )


def stable_hash(values: pd.Series) -> str:
    payload = ",".join(str(int(v)) for v in sorted(values.astype(int).tolist()))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def load_unsw_split(
    dataset_csv: str,
    split_seed: int,
    split_index: str,
    tier: str,
    feature_mode: str,
):
    csv_path = rp(dataset_csv)
    df = pd.read_csv(csv_path)
    label2id = {v: k for k, v in LABEL_MAP.items()}
    if "attack_cat" not in df.columns:
        raise RuntimeError("Column 'attack_cat' not found in curated UNSW CSV.")
    df["source_row_id"] = np.arange(len(df), dtype=np.int64)
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    df["label"] = df["attack_cat"].map(label2id)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map(LABEL_MAP)

    split_path = split_index.format(seed=split_seed) if "{seed}" in split_index else split_index
    split_path = rp(split_path)
    if not os.path.exists(split_path):
        raise RuntimeError(f"Split index not found: {split_path}")

    idx = np.load(split_path)
    train_df = df.iloc[idx["train_idx"]].copy()
    val_df = df.iloc[idx["val_idx"]].copy()

    if tier in PER_CLASS_CAP:
        train_df = stratified_cap(train_df, PER_CLASS_CAP[tier], split_seed)
        val_df = stratified_cap(val_df, PER_CLASS_CAP[tier], split_seed + 1)

    cat_cols, num_cols = infer_feature_columns(train_df, feature_mode)
    feature_cols = cat_cols + num_cols
    if num_cols:
        train_df[num_cols] = train_df[num_cols].replace([np.inf, -np.inf], np.nan)
        val_df[num_cols] = val_df[num_cols].replace([np.inf, -np.inf], np.nan)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), cat_cols, num_cols, feature_cols


def build_val_variant(df: pd.DataFrame, num_cols: List[str], experiment: str, seed: int) -> pd.DataFrame:
    if experiment == "E1_clean":
        return df.copy()
    if experiment.startswith("E3_mask_"):
        ratio = float(experiment.split("_")[-1]) / 100.0
        return apply_missing_mask(df, num_cols, ratio, seed=seed + 100)
    if experiment.startswith("E3_block_"):
        ratio = float(experiment.split("_")[-1]) / 100.0
        return apply_block_missing_mask(df, infer_block_groups(num_cols), ratio, seed=seed + 150)
    if experiment.startswith("E4_noise_"):
        ratio = float(experiment.split("_")[-1]) / 100.0
        return apply_noise(df, num_cols, ratio, seed=seed + 200)
    raise ValueError(f"Unsupported experiment '{experiment}'.")


def export_frame(path: str, df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> None:
    out = df[["source_row_id", "label", "label_name", "attack_cat"] + cat_cols + num_cols].copy()
    for col in num_cols:
        if f"{col}_cat" in df.columns:
            out[f"{col}_cat"] = df[f"{col}_cat"]
    out["text"] = build_text(df, cat_cols, num_cols)
    ensure_dir(os.path.dirname(path))
    out.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    split_seeds = [int(x) for x in parse_csv_list(args.split_seeds)]
    experiments = parse_csv_list(args.val_experiments)
    output_dir = rp(args.output_dir)
    ensure_dir(output_dir)

    manifests = []
    for split_seed in split_seeds:
        train_df, val_df, cat_cols, num_cols, feature_cols = load_unsw_split(
            dataset_csv=args.dataset_csv,
            split_seed=split_seed,
            split_index=args.split_index,
            tier=args.tier,
            feature_mode=args.feature_mode,
        )

        seed_dir = os.path.join(output_dir, f"seed{split_seed}")
        ensure_dir(seed_dir)

        bin_specs = {}
        for col in num_cols:
            bins, labels = fit_bins(train_df[col], max_bins=len(BIN_LABELS))
            train_df[f"{col}_cat"] = apply_bins(train_df[col], bins, labels)
            bin_specs[col] = {
                "bins": None if bins is None else [float(x) for x in bins.tolist()],
                "labels": labels,
            }

        train_path = os.path.join(
            seed_dir,
            f"{args.data_prefix}_train_{args.tier}_E1_clean.csv",
        )
        export_frame(train_path, train_df, cat_cols, num_cols)

        for experiment in experiments:
            val_variant = build_val_variant(val_df, num_cols, experiment, split_seed)
            for col in num_cols:
                spec = bin_specs[col]
                bins = None if spec["bins"] is None else np.asarray(spec["bins"], dtype=np.float64)
                val_variant[f"{col}_cat"] = apply_bins(val_variant[col], bins, spec["labels"])
            val_path = os.path.join(
                seed_dir,
                f"{args.data_prefix}_val_{args.tier}_{experiment}.csv",
            )
            export_frame(val_path, val_variant, cat_cols, num_cols)

        manifest = {
            "dataset": "UNSW-NB15",
            "tier": args.tier,
            "split_seed": split_seed,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "train_class_counts": {LABEL_MAP[int(k)]: int(v) for k, v in train_df["label"].value_counts().sort_index().items()},
            "val_class_counts": {LABEL_MAP[int(k)]: int(v) for k, v in val_df["label"].value_counts().sort_index().items()},
            "feature_cols": feature_cols,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "text_numeric_bins": bin_specs,
            "block_groups": infer_block_groups(num_cols),
            "train_source_hash": stable_hash(train_df["source_row_id"]),
            "val_source_hash": stable_hash(val_df["source_row_id"]),
            "train_csv": train_path,
        }
        manifest_path = os.path.join(seed_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        manifests.append(manifest)
        print(f"[FAIR BUNDLE] seed={split_seed} -> {seed_dir}")

    merged_manifest = {
        "dataset": "UNSW-NB15",
        "tier": args.tier,
        "feature_mode": args.feature_mode,
        "split_seeds": split_seeds,
        "experiments": experiments,
        "manifests": manifests,
    }
    summary_path = os.path.join(output_dir, f"{args.data_prefix}_{args.tier}_manifest.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(merged_manifest, f, ensure_ascii=False, indent=2)
    print(f"[SUMMARY] {summary_path}")


if __name__ == "__main__":
    main()
