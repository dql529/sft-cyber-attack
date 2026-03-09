from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robustness_utils import ensure_dir, rp


TIER_CAPS = {"mini": 300, "medium": 3000}
BENIGN_LABEL = 0
ATTACK_LABEL = 1
LABEL_NAME_MAP = {0: "benign", 1: "attack"}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build a unified binary text benchmark bundle from UNSW, CICIDS2017, and UAV datasets."
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--output-dir", default="./data/binary_prompt_csv")
    ap.add_argument("--chunksize", type=int, default=50000)
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


def build_text(row: Dict[str, object], feature_cols: List[str], renamed: Dict[str, str]) -> str:
    parts = []
    for col in feature_cols:
        parts.append(f"{renamed[col]}={normalize_text_value(row.get(col))}")
    return " ; ".join(parts)


def normalize_attack_cat(x: object) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    return re.sub(r"[^\w\s-]+$", "", s).strip()


@dataclass
class Reservoir:
    cap: int
    seed: int
    items: List[Dict] = field(default_factory=list)
    seen: int = 0

    def consider(self, row: Dict) -> None:
        self.seen += 1
        if len(self.items) < self.cap:
            self.items.append(row)
            return
        rng = np.random.default_rng(self.seed + self.seen)
        j = int(rng.integers(0, self.seen))
        if j < self.cap:
            self.items[j] = row


@dataclass
class DatasetAccumulator:
    dataset_name: str
    output_prefix: str
    seed: int
    val_ratio: float
    train_reservoirs: Dict[int, Reservoir] = field(init=False)
    val_reservoirs: Dict[int, Reservoir] = field(init=False)
    total_rows: int = 0
    split_counts: Counter = field(default_factory=Counter)
    label_counts: Counter = field(default_factory=Counter)
    source_label_counts: Counter = field(default_factory=Counter)

    def __post_init__(self):
        self.train_reservoirs = {
            BENIGN_LABEL: Reservoir(TIER_CAPS["medium"], self.seed + 11),
            ATTACK_LABEL: Reservoir(TIER_CAPS["medium"], self.seed + 19),
        }
        self.val_reservoirs = {
            BENIGN_LABEL: Reservoir(TIER_CAPS["medium"], self.seed + 23),
            ATTACK_LABEL: Reservoir(TIER_CAPS["medium"], self.seed + 29),
        }
        self.rng = np.random.default_rng(self.seed)

    def add_row(self, row: Dict, label: int, source_label: str) -> None:
        split = "val" if self.rng.random() < self.val_ratio else "train"
        payload = {
            "dataset_name": self.dataset_name,
            "source_row_id": row["source_row_id"],
            "source_file": row["source_file"],
            "source_label": source_label,
            "label_name": LABEL_NAME_MAP[label],
            "label": int(label),
            "text": row["text"],
        }
        self.total_rows += 1
        self.label_counts[LABEL_NAME_MAP[label]] += 1
        self.split_counts[f"{split}_{LABEL_NAME_MAP[label]}"] += 1
        self.source_label_counts[source_label] += 1
        if split == "train":
            self.train_reservoirs[label].consider(payload)
        else:
            self.val_reservoirs[label].consider(payload)

    def tier_frame(self, split: str, tier: str) -> pd.DataFrame:
        if tier not in TIER_CAPS:
            raise ValueError(f"Unsupported tier '{tier}'.")
        stable_offset = sum(ord(ch) for ch in f"{self.dataset_name}:{split}:{tier}")
        rng = np.random.default_rng(self.seed + stable_offset)
        reservoirs = self.train_reservoirs if split == "train" else self.val_reservoirs
        parts = []
        for label in [BENIGN_LABEL, ATTACK_LABEL]:
            items = reservoirs[label].items
            if not items:
                continue
            if tier == "medium" or len(items) <= TIER_CAPS[tier]:
                parts.append(pd.DataFrame(items))
            else:
                idx = rng.choice(len(items), size=TIER_CAPS[tier], replace=False)
                parts.append(pd.DataFrame([items[i] for i in idx]))
        if not parts:
            return pd.DataFrame(
                columns=[
                    "dataset_name",
                    "source_row_id",
                    "source_file",
                    "source_label",
                    "label_name",
                    "label",
                    "text",
                ]
            )
        df = pd.concat(parts, ignore_index=True)
        return df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

    def summary_rows(self) -> List[Dict]:
        rows = []
        for tier in sorted(TIER_CAPS):
            for split in ["train", "val"]:
                df = self.tier_frame(split, tier)
                counts = df["label_name"].value_counts().to_dict() if not df.empty else {}
                rows.append(
                    {
                        "dataset_name": self.dataset_name,
                        "output_prefix": self.output_prefix,
                        "tier": tier,
                        "split": split,
                        "rows": int(len(df)),
                        "benign_rows": int(counts.get("benign", 0)),
                        "attack_rows": int(counts.get("attack", 0)),
                        "source_total_rows": int(self.total_rows),
                    }
                )
        return rows


def iter_unsw_rows(csv_path: str, dataset_name: str) -> Iterator[Dict]:
    df = pd.read_csv(rp(csv_path), low_memory=False)
    label_col = "attack_cat"
    feature_cols = [c for c in df.columns if c != label_col]
    renamed = {c: slugify_col(c) for c in feature_cols}
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        row_dict = row._asdict()
        source_label = normalize_attack_cat(row_dict.get(label_col))
        if not source_label:
            continue
        label = BENIGN_LABEL if source_label == "Normal" else ATTACK_LABEL
        yield {
            "source_row_id": f"{dataset_name}:{idx}",
            "source_file": os.path.basename(csv_path),
            "source_label": source_label,
            "label": label,
            "text": build_text(row_dict, feature_cols, renamed),
        }


def iter_uav_rows(csv_path: str, dataset_name: str) -> Iterator[Dict]:
    df = pd.read_csv(rp(csv_path), low_memory=False)
    label_col = "class"
    feature_cols = [c for c in df.columns if c != label_col]
    renamed = {c: slugify_col(c) for c in feature_cols}
    records = df.to_dict(orient="records")
    for idx, row_dict in enumerate(records, start=1):
        source_label = str(row_dict.get(label_col)).strip()
        if not source_label or source_label.lower() in {"nan", "class"}:
            continue
        label = BENIGN_LABEL if source_label.lower() == "benign" else ATTACK_LABEL
        yield {
            "source_row_id": f"{dataset_name}:{idx}",
            "source_file": os.path.basename(csv_path),
            "source_label": source_label,
            "label": label,
            "text": build_text(row_dict, feature_cols, renamed),
        }


def iter_cic_rows(csv_dir: str, dataset_name: str, chunksize: int) -> Iterator[Dict]:
    csv_dir = rp(csv_dir)
    files = sorted(
        [
            os.path.join(csv_dir, name)
            for name in os.listdir(csv_dir)
            if name.lower().endswith(".csv") and name != "README.md"
        ]
    )
    for csv_path in files:
        row_offset = 0
        for chunk in pd.read_csv(
            csv_path,
            low_memory=False,
            chunksize=chunksize,
            encoding_errors="ignore",
        ):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            label_col = "Label"
            if label_col not in chunk.columns:
                if " Label" in chunk.columns:
                    label_col = " Label"
                else:
                    raise RuntimeError(f"Could not find label column in {csv_path}")
            feature_cols = [c for c in chunk.columns if c != label_col]
            renamed = {c: slugify_col(c) for c in feature_cols}
            records = chunk.to_dict(orient="records")
            for local_idx, row_dict in enumerate(records, start=1):
                source_label = str(row_dict.get(label_col)).strip()
                if not source_label or source_label.lower() == "label":
                    continue
                label = BENIGN_LABEL if source_label.upper() == "BENIGN" else ATTACK_LABEL
                yield {
                    "source_row_id": f"{dataset_name}:{os.path.basename(csv_path)}:{row_offset + local_idx}",
                    "source_file": os.path.basename(csv_path),
                    "source_label": source_label,
                    "label": label,
                    "text": build_text(row_dict, feature_cols, renamed),
                }
            row_offset += len(records)


def write_bundle(acc: DatasetAccumulator, output_dir: str) -> List[Dict]:
    ensure_dir(rp(output_dir))
    for tier in sorted(TIER_CAPS):
        for split in ["train", "val"]:
            df = acc.tier_frame(split, tier)
            out_path = os.path.join(rp(output_dir), f"{acc.output_prefix}_{split}_{tier}_E1_clean.csv")
            df.to_csv(out_path, index=False)
    return acc.summary_rows()


def main():
    args = parse_args()
    output_dir = rp(args.output_dir)
    ensure_dir(output_dir)

    sources = [
        (
            "unsw_nb15_curated_binary",
            "unsw_curated_binary",
            iter_unsw_rows(r"./data/datasets/unsw_nb15/curated/unsw15_filtered_nolog.csv", "unsw_nb15_curated_binary"),
        ),
        (
            "unsw_nb15_raw_binary",
            "unsw_raw_binary",
            iter_unsw_rows(r"./data/datasets/unsw_nb15/raw/Augmented-UNSW_NB15.csv", "unsw_nb15_raw_binary"),
        ),
        (
            "uav_cyberattacks_binary",
            "uav_binary",
            iter_uav_rows(r"./data/datasets/uav_cyberattacks/raw/Dataset_T-ITS.csv", "uav_cyberattacks_binary"),
        ),
        (
            "cicids2017_binary",
            "cicids2017_binary",
            iter_cic_rows(r"./data/datasets/cicids2017/raw", "cicids2017_binary", args.chunksize),
        ),
    ]

    summary_rows = []
    cards = []
    for dataset_name, output_prefix, iterator in sources:
        print(f"[BUILD] {dataset_name}")
        acc = DatasetAccumulator(
            dataset_name=dataset_name,
            output_prefix=output_prefix,
            seed=args.seed,
            val_ratio=args.val_ratio,
        )
        for row in iterator:
            acc.add_row(row, row["label"], row["source_label"])
        summary_rows.extend(write_bundle(acc, output_dir))
        cards.append(
            {
                "dataset_name": dataset_name,
                "output_prefix": output_prefix,
                "total_rows": acc.total_rows,
                "label_counts": dict(acc.label_counts),
                "split_counts": dict(acc.split_counts),
                "top_source_labels": dict(acc.source_label_counts.most_common(20)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "binary_dataset_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    manifest = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "label_map": LABEL_NAME_MAP,
        "datasets": cards,
    }
    manifest_path = os.path.join(output_dir, "binary_dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[SUMMARY] {summary_path}")
    print(f"[MANIFEST] {manifest_path}")


if __name__ == "__main__":
    main()
