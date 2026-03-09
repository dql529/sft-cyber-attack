from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nn_tabular_baseline import (
    apply_block_missing_mask,
    apply_missing_mask,
    apply_noise,
    infer_block_groups,
)
from robustness_utils import ensure_dir, rp


BENIGN_LABEL = 0
ATTACK_LABEL = 1
LABEL_NAME_MAP = {0: "benign", 1: "attack"}
MEDIUM_CAP = 3000
DEFAULT_EXPERIMENTS = [
    "E1_clean",
    "E3_mask_30",
    "E3_mask_50",
    "E3_block_30",
    "E3_block_50",
    "E4_noise_30",
    "E4_noise_50",
]
METADATA_COLS = ["dataset_name", "source_row_id", "source_file", "source_label", "label_name", "label", "text"]


def parse_args():
    ap = argparse.ArgumentParser(description="Build a CICIDS2017 binary structured/text bundle for fast validation.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--chunksize", type=int, default=50000)
    ap.add_argument("--output-dir", default="./data/cic_binary_structured")
    ap.add_argument("--data-prefix", default="cic_binary_structured")
    ap.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS))
    return ap.parse_args()


def slugify_col(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "col"


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
    s = re.sub(r"\s+", "_", s)
    return s


def build_text(df: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    return df[feature_cols].apply(
        lambda row: " ; ".join(f"{col}={normalize_text_value(row[col])}" for col in feature_cols),
        axis=1,
    )


@dataclass
class Reservoir:
    cap: int
    seed: int
    rows: List[Dict] = field(default_factory=list)
    seen: int = 0

    def consider(self, row: Dict) -> None:
        self.seen += 1
        if len(self.rows) < self.cap:
            self.rows.append(row)
            return
        rng = np.random.default_rng(self.seed + self.seen)
        idx = int(rng.integers(0, self.seen))
        if idx < self.cap:
            self.rows[idx] = row


def iter_cic_rows(csv_dir: str, chunksize: int):
    csv_dir = rp(csv_dir)
    files = sorted(
        [
            os.path.join(csv_dir, name)
            for name in os.listdir(csv_dir)
            if name.lower().endswith(".csv")
        ]
    )
    for csv_path in files:
        row_offset = 0
        for chunk in pd.read_csv(csv_path, low_memory=False, chunksize=chunksize, encoding_errors="ignore"):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            label_col = "Label" if "Label" in chunk.columns else " Label"
            feature_cols = [c for c in chunk.columns if c != label_col]
            renamed = {c: slugify_col(c) for c in feature_cols}
            records = chunk.to_dict(orient="records")
            for local_idx, row in enumerate(records, start=1):
                source_label = str(row.get(label_col)).strip()
                if not source_label or source_label.lower() == "label":
                    continue
                label = BENIGN_LABEL if source_label.upper() == "BENIGN" else ATTACK_LABEL
                payload = {
                    "dataset_name": "cicids2017_binary",
                    "source_row_id": f"cicids2017_binary:{os.path.basename(csv_path)}:{row_offset + local_idx}",
                    "source_file": os.path.basename(csv_path),
                    "source_label": source_label,
                    "label_name": LABEL_NAME_MAP[label],
                    "label": int(label),
                }
                for col in feature_cols:
                    payload[renamed[col]] = row.get(col)
                yield payload
            row_offset += len(records)


def coerce_feature_types(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str]):
    cat_cols = []
    num_cols = []
    for col in feature_cols:
        tr_num = pd.to_numeric(train_df[col], errors="coerce")
        va_num = pd.to_numeric(val_df[col], errors="coerce")
        tr_ratio = float(tr_num.notna().mean()) if len(tr_num) else 0.0
        va_ratio = float(va_num.notna().mean()) if len(va_num) else 0.0
        if min(tr_ratio, va_ratio) >= 0.95:
            train_df[col] = tr_num.replace([np.inf, -np.inf], np.nan)
            val_df[col] = va_num.replace([np.inf, -np.inf], np.nan)
            num_cols.append(col)
        else:
            train_df[col] = train_df[col].astype(str)
            val_df[col] = val_df[col].astype(str)
            cat_cols.append(col)
    return train_df, val_df, cat_cols, num_cols


def make_variant(df: pd.DataFrame, num_cols: List[str], experiment: str, seed: int):
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


def main():
    args = parse_args()
    output_dir = rp(args.output_dir)
    ensure_dir(output_dir)
    experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]

    rng = np.random.default_rng(args.seed)
    train_res = {
        BENIGN_LABEL: Reservoir(MEDIUM_CAP, args.seed + 11),
        ATTACK_LABEL: Reservoir(MEDIUM_CAP, args.seed + 19),
    }
    val_res = {
        BENIGN_LABEL: Reservoir(MEDIUM_CAP, args.seed + 23),
        ATTACK_LABEL: Reservoir(MEDIUM_CAP, args.seed + 29),
    }
    total_rows = 0
    label_counts = {LABEL_NAME_MAP[BENIGN_LABEL]: 0, LABEL_NAME_MAP[ATTACK_LABEL]: 0}

    for row in iter_cic_rows("./data/datasets/cicids2017/raw", args.chunksize):
        total_rows += 1
        label_counts[row["label_name"]] += 1
        split = "val" if rng.random() < args.val_ratio else "train"
        if split == "train":
            train_res[row["label"]].consider(row)
        else:
            val_res[row["label"]].consider(row)

    train_df = pd.concat([pd.DataFrame(train_res[0].rows), pd.DataFrame(train_res[1].rows)], ignore_index=True)
    val_df = pd.concat([pd.DataFrame(val_res[0].rows), pd.DataFrame(val_res[1].rows)], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=args.seed + 1).reset_index(drop=True)

    feature_cols = [c for c in train_df.columns if c not in METADATA_COLS]
    train_df, val_df, cat_cols, num_cols = coerce_feature_types(train_df, val_df, feature_cols)
    train_df["text"] = build_text(train_df, feature_cols)
    val_df["text"] = build_text(val_df, feature_cols)

    train_path = os.path.join(output_dir, f"{args.data_prefix}_train_medium_E1_clean.csv")
    val_clean_path = os.path.join(output_dir, f"{args.data_prefix}_val_medium_E1_clean.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_clean_path, index=False)

    for experiment in experiments:
        variant = make_variant(val_df.drop(columns=["text"]), num_cols, experiment, args.seed)
        variant["text"] = build_text(variant, feature_cols)
        out_path = os.path.join(output_dir, f"{args.data_prefix}_val_medium_{experiment}.csv")
        variant.to_csv(out_path, index=False)

    manifest = {
        "dataset_name": "cicids2017_binary",
        "data_prefix": args.data_prefix,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "total_rows": total_rows,
        "label_counts": label_counts,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "experiments": experiments,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[TRAIN] {train_path}")
    print(f"[VAL] {val_clean_path}")
    print(f"[MANIFEST] {os.path.join(output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
