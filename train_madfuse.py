from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robustness_utils import (
    DEFAULT_VAL_EXPERIMENTS,
    compute_classification_metrics,
    count_torch_parameters,
    ensure_dir,
    flatten_metric_record,
    infer_prompt_paths,
    mean_latency_ms,
    parse_csv_list,
    parse_experiment_metadata,
    parse_int_list,
    peak_memory_mb,
    reset_peak_memory,
    rp,
    save_json,
    set_seed,
)
from config import LABEL_MAP


TEXT_MODEL_PATH = "./models/bert-base-uncased"
CAT_COLS = ["proto", "service", "state"]
NUM_COLS = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "sload", "dload", "tcprtt", "ct_srv_src"]
BIN_TARGET_VOCAB = ["missing", "zero", "positive", "low", "medium", "high", "very_high"]
BLOCK_GROUPS = [
    ["dur", "spkts", "dpkts", "sbytes", "dbytes"],
    ["sload", "dload", "tcprtt", "ct_srv_src"],
]
BIN_TARGET_TO_ID = {name: idx for idx, name in enumerate(BIN_TARGET_VOCAB)}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-dir", default="./data/prompt_csv")
    ap.add_argument("--data-prefix", default="unsw15_prompt")
    ap.add_argument("--dataset-name", default="unsw15")
    ap.add_argument("--model-path", default=TEXT_MODEL_PATH)
    ap.add_argument("--tier", choices=["mini", "medium", "full"], default="medium")
    ap.add_argument("--train-experiment", default="E1_clean")
    ap.add_argument("--val-experiments", default=",".join(DEFAULT_VAL_EXPERIMENTS))
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--seeds", default="42,52,62")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--train-batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--cat-embed-dim", type=int, default=32)
    ap.add_argument("--cons-weight", type=float, default=0.3)
    ap.add_argument("--rec-weight", type=float, default=0.2)
    ap.add_argument("--branch-dropout-p", type=float, default=0.2)
    ap.add_argument("--mask-ratios", default="0,0.1,0.3,0.5")
    ap.add_argument("--mask-modes", default="iid,block")
    ap.add_argument("--freeze-text-encoder", action="store_true")
    ap.add_argument("--no-mask-branch", action="store_true")
    ap.add_argument("--no-fusion-gate", action="store_true")
    ap.add_argument("--no-consistency-loss", action="store_true")
    ap.add_argument("--no-reconstruction-loss", action="store_true")
    ap.add_argument("--no-branch-dropout", action="store_true")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--tabular-only", action="store_true")
    ap.add_argument("--output-root", default="./madfuse_outputs")
    return ap.parse_args()


def required_columns(text_col: str) -> List[str]:
    return (
        ["source_row_id", "label", text_col, "num_missing_ratio", "num_observed_ratio"]
        + CAT_COLS
        + NUM_COLS
        + [f"{col}_cat" for col in NUM_COLS]
        + [f"obs_{col}" for col in CAT_COLS + NUM_COLS]
    )


def validate_frame(df: pd.DataFrame, text_col: str, csv_path: str) -> None:
    missing = sorted(set(required_columns(text_col)) - set(df.columns))
    if missing:
        raise RuntimeError(
            f"{csv_path} is missing required columns for MADFUSE: {missing}. "
            "Re-run prepare.py after the unified export update."
        )


def build_text_from_bins(proto: str, state: str, service: str, bin_labels: List[str]) -> str:
    parts = [f"proto={proto}", f"state={state}", f"service={service}"]
    for col, label in zip(NUM_COLS, bin_labels):
        parts.append(f"{col}={label}")
    return " ; ".join(parts)


def fit_category_vocab(train_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    vocabs: Dict[str, Dict[str, int]] = {}
    for col in CAT_COLS:
        values = train_df[col].astype("string").fillna("UNK").tolist()
        uniq = ["UNK"] + sorted({str(v) for v in values if str(v) != "UNK"})
        vocabs[col] = {token: idx for idx, token in enumerate(uniq)}
    return vocabs


def fit_numeric_stats(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in NUM_COLS:
        values = pd.to_numeric(train_df[col], errors="coerce")
        obs = train_df[f"obs_{col}"].astype(float) > 0.5
        observed = values[obs & values.notna()]
        if observed.empty:
            mean, std = 0.0, 1.0
        else:
            mean = float(observed.mean())
            std = float(observed.std(ddof=0))
            if std < 1e-6:
                std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def category_id(vocab: Dict[str, int], value: str) -> int:
    token = str(value) if pd.notna(value) else "UNK"
    return vocab.get(token, vocab["UNK"])


def encode_numeric(values: np.ndarray, obs: np.ndarray, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    out = np.zeros(len(NUM_COLS), dtype=np.float32)
    for idx, col in enumerate(NUM_COLS):
        if obs[idx] < 0.5 or not np.isfinite(values[idx]):
            out[idx] = 0.0
            continue
        out[idx] = float((values[idx] - stats[col]["mean"]) / stats[col]["std"])
    return out


def build_records(df: pd.DataFrame, text_col: str, cat_vocab, num_stats) -> List[Dict]:
    records = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        proto = str(row_dict["proto"])
        state = str(row_dict["state"])
        service = str(row_dict["service"])
        num_values = np.array(
            [float(row_dict[col]) if pd.notna(row_dict[col]) else np.nan for col in NUM_COLS],
            dtype=np.float32,
        )
        num_obs = np.array([float(row_dict[f"obs_{col}"]) for col in NUM_COLS], dtype=np.float32)
        cat_obs = np.array([float(row_dict[f"obs_{col}"]) for col in CAT_COLS], dtype=np.float32)
        num_scaled = encode_numeric(num_values, num_obs, num_stats)
        bin_labels = [str(row_dict[f"{col}_cat"]) for col in NUM_COLS]
        bin_ids = np.array([BIN_TARGET_TO_ID.get(label, 0) for label in bin_labels], dtype=np.int64)
        mask_features = np.concatenate(
            [
                cat_obs,
                num_obs,
                np.array(
                    [float(row_dict["num_missing_ratio"]), float(row_dict["num_observed_ratio"])],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        records.append(
            {
                "source_row_id": int(row_dict["source_row_id"]),
                "label": int(row_dict["label"]),
                "text": str(row_dict[text_col]),
                "proto": proto,
                "state": state,
                "service": service,
                "cat_ids": np.array(
                    [category_id(cat_vocab[col], row_dict[col]) for col in CAT_COLS],
                    dtype=np.int64,
                ),
                "cat_obs": cat_obs,
                "num_values": num_values,
                "num_scaled": num_scaled,
                "num_obs": num_obs,
                "num_bin_labels": bin_labels,
                "bin_ids": bin_ids,
                "mask_features": mask_features,
            }
        )
    return records


@dataclass
class AugmentConfig:
    mask_ratios: List[float]
    mask_modes: List[str]


def block_indices() -> List[List[int]]:
    return [[NUM_COLS.index(col) for col in group] for group in BLOCK_GROUPS]


BLOCK_INDICES = block_indices()


def sample_augmented_view(sample: Dict, cfg: AugmentConfig, rng: np.random.Generator) -> Dict:
    ratio = cfg.mask_ratios[int(rng.integers(0, len(cfg.mask_ratios)))]
    mode = cfg.mask_modes[int(rng.integers(0, len(cfg.mask_modes)))]
    aug_num_obs = sample["num_obs"].copy()
    masked_positions = np.zeros(len(NUM_COLS), dtype=bool)

    if ratio > 0:
        observed_positions = sample["num_obs"] > 0.5
        if mode == "iid":
            sampled = rng.random(len(NUM_COLS)) < ratio
            masked_positions = sampled & observed_positions
        elif mode == "block":
            for block in BLOCK_INDICES:
                if rng.random() < ratio:
                    for idx in block:
                        if observed_positions[idx]:
                            masked_positions[idx] = True
        else:
            raise ValueError(f"Unsupported mask mode '{mode}'.")

    aug_num_obs[masked_positions] = 0.0
    aug_num_scaled = sample["num_scaled"].copy()
    aug_num_scaled[masked_positions] = 0.0
    aug_bin_labels = list(sample["num_bin_labels"])
    for idx in np.where(masked_positions)[0]:
        aug_bin_labels[idx] = "missing"
    aug_mask_features = np.concatenate(
        [
            sample["cat_obs"],
            aug_num_obs,
            np.array([1.0 - aug_num_obs.mean(), aug_num_obs.mean()], dtype=np.float32),
        ]
    ).astype(np.float32)
    aug_text = build_text_from_bins(sample["proto"], sample["state"], sample["service"], aug_bin_labels)
    return {
        "text": aug_text,
        "num_scaled": aug_num_scaled.astype(np.float32),
        "num_obs": aug_num_obs.astype(np.float32),
        "mask_features": aug_mask_features.astype(np.float32),
        "rec_mask": masked_positions.astype(np.float32),
    }


class RecordDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


class TrainCollator:
    def __init__(self, tokenizer, max_len: int, augment_cfg: AugmentConfig):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment_cfg = augment_cfg

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng()
        labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
        clean_texts = [sample["text"] for sample in batch]
        aug_views = [sample_augmented_view(sample, self.augment_cfg, rng) for sample in batch]
        aug_texts = [view["text"] for view in aug_views]
        clean_tok = self.tokenizer(
            clean_texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        aug_tok = self.tokenizer(
            aug_texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "labels": labels,
            "clean_input_ids": clean_tok["input_ids"],
            "clean_attention_mask": clean_tok["attention_mask"],
            "aug_input_ids": aug_tok["input_ids"],
            "aug_attention_mask": aug_tok["attention_mask"],
            "cat_ids": torch.tensor(np.stack([sample["cat_ids"] for sample in batch]), dtype=torch.long),
            "clean_num_scaled": torch.tensor(
                np.stack([sample["num_scaled"] for sample in batch]), dtype=torch.float32
            ),
            "clean_num_obs": torch.tensor(
                np.stack([sample["num_obs"] for sample in batch]), dtype=torch.float32
            ),
            "clean_mask_features": torch.tensor(
                np.stack([sample["mask_features"] for sample in batch]), dtype=torch.float32
            ),
            "aug_num_scaled": torch.tensor(
                np.stack([view["num_scaled"] for view in aug_views]), dtype=torch.float32
            ),
            "aug_num_obs": torch.tensor(
                np.stack([view["num_obs"] for view in aug_views]), dtype=torch.float32
            ),
            "aug_mask_features": torch.tensor(
                np.stack([view["mask_features"] for view in aug_views]), dtype=torch.float32
            ),
            "rec_targets": torch.tensor(np.stack([sample["bin_ids"] for sample in batch]), dtype=torch.long),
            "rec_mask": torch.tensor(np.stack([view["rec_mask"] for view in aug_views]), dtype=torch.float32),
        }


class EvalCollator:
    def __init__(self, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [sample["text"] for sample in batch]
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "labels": torch.tensor([sample["label"] for sample in batch], dtype=torch.long),
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "cat_ids": torch.tensor(np.stack([sample["cat_ids"] for sample in batch]), dtype=torch.long),
            "num_scaled": torch.tensor(np.stack([sample["num_scaled"] for sample in batch]), dtype=torch.float32),
            "num_obs": torch.tensor(np.stack([sample["num_obs"] for sample in batch]), dtype=torch.float32),
            "mask_features": torch.tensor(
                np.stack([sample["mask_features"] for sample in batch]), dtype=torch.float32
            ),
        }


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MadfuseModel(nn.Module):
    def __init__(self, model_path: str, cat_vocab: Dict[str, Dict[str, int]], args):
        super().__init__()
        if AutoModel is None:
            raise RuntimeError("transformers is required to run MADFUSE-IDS.")
        self.args = args
        self.freeze_text_encoder = bool(args.freeze_text_encoder)
        self.use_mask_branch = not args.no_mask_branch
        self.use_fusion_gate = not args.no_fusion_gate
        self.use_branch_dropout = not args.no_branch_dropout
        self.text_only = bool(args.text_only)
        self.tabular_only = bool(args.tabular_only)
        self.text_encoder = AutoModel.from_pretrained(rp(model_path), local_files_only=True)
        if self.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        text_hidden = int(getattr(self.text_encoder.config, "hidden_size"))
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
            nn.Dropout(args.dropout),
        )
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(len(cat_vocab[col]), args.cat_embed_dim) for col in CAT_COLS]
        )
        tab_in_dim = len(CAT_COLS) * args.cat_embed_dim + len(NUM_COLS) * 2
        self.tabular_mlp = MLP(tab_in_dim, args.hidden_dim, args.hidden_dim, args.dropout)
        mask_in_dim = len(CAT_COLS) + len(NUM_COLS) + 2
        self.mask_mlp = MLP(mask_in_dim, args.hidden_dim, args.hidden_dim, args.dropout)
        fusion_in_dim = args.hidden_dim * 3
        self.gate = nn.Sequential(
            nn.Linear(fusion_in_dim, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_proj = MLP(fusion_in_dim, args.hidden_dim, args.hidden_dim, args.dropout)
        self.classifier = nn.Linear(args.hidden_dim, len(LABEL_MAP))
        self.rec_heads = nn.ModuleList(
            [nn.Linear(args.hidden_dim, len(BIN_TARGET_VOCAB)) for _ in NUM_COLS]
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        return self

    def encode_text(self, input_ids, attention_mask):
        ctx = torch.no_grad() if self.freeze_text_encoder else nullcontext()
        with ctx:
            out = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.text_proj(pooled)

    def encode_tabular(self, cat_ids, num_scaled, num_obs):
        cat_chunks = [emb(cat_ids[:, idx]) for idx, emb in enumerate(self.cat_embeddings)]
        tab_input = torch.cat(cat_chunks + [num_scaled, num_obs], dim=1)
        return self.tabular_mlp(tab_input)

    def encode_mask(self, mask_features):
        if not self.use_mask_branch:
            return torch.zeros(mask_features.shape[0], self.args.hidden_dim, device=mask_features.device)
        return self.mask_mlp(mask_features)

    def apply_branch_dropout(self, z_text, z_tab):
        if not self.training or not self.use_branch_dropout or self.args.branch_dropout_p <= 0:
            return z_text, z_tab
        selector = torch.rand(z_text.shape[0], device=z_text.device)
        drop_any = selector < self.args.branch_dropout_p
        drop_text = drop_any & (torch.rand_like(selector) < 0.5)
        drop_tab = drop_any & (~drop_text)
        z_text = z_text.masked_fill(drop_text.unsqueeze(-1), 0.0)
        z_tab = z_tab.masked_fill(drop_tab.unsqueeze(-1), 0.0)
        return z_text, z_tab

    def fuse(self, z_text, z_tab, z_mask):
        if self.text_only:
            return z_text
        if self.tabular_only:
            return z_tab + z_mask
        z_text, z_tab = self.apply_branch_dropout(z_text, z_tab)
        fusion_input = torch.cat([z_text, z_tab, z_mask], dim=1)
        if self.use_fusion_gate:
            gate = self.gate(fusion_input)
            return gate * z_tab + (1.0 - gate) * z_text + z_mask
        return self.fusion_proj(fusion_input)

    def forward_single(self, input_ids, attention_mask, cat_ids, num_scaled, num_obs, mask_features):
        z_text = self.encode_text(input_ids, attention_mask)
        z_tab = self.encode_tabular(cat_ids, num_scaled, num_obs)
        z_mask = self.encode_mask(mask_features)
        z = self.fuse(z_text, z_tab, z_mask)
        logits = self.classifier(z)
        rec_logits = [head(z) for head in self.rec_heads]
        return {"logits": logits, "rec_logits": rec_logits}


def move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def js_divergence(logits_a, logits_b) -> torch.Tensor:
    p = F.softmax(logits_a, dim=-1)
    q = F.softmax(logits_b, dim=-1)
    m = 0.5 * (p + q)
    return 0.5 * (
        F.kl_div(F.log_softmax(logits_a, dim=-1), m, reduction="batchmean")
        + F.kl_div(F.log_softmax(logits_b, dim=-1), m, reduction="batchmean")
    )


def reconstruction_loss(rec_logits, rec_targets, rec_mask) -> torch.Tensor:
    losses = []
    for feat_idx, head_logits in enumerate(rec_logits):
        mask = rec_mask[:, feat_idx] > 0.5
        if mask.any():
            losses.append(F.cross_entropy(head_logits[mask], rec_targets[mask, feat_idx]))
    if not losses:
        return torch.zeros((), device=rec_targets.device)
    return torch.stack(losses).mean()


def model_variant_name(args) -> str:
    parts = ["MADFUSE-IDS"]
    if args.no_mask_branch:
        parts.append("no_mask_branch")
    if args.no_fusion_gate:
        parts.append("no_fusion_gate")
    if args.no_consistency_loss:
        parts.append("no_cons")
    if args.no_reconstruction_loss:
        parts.append("no_rec")
    if args.no_branch_dropout:
        parts.append("no_branch_dropout")
    if args.text_only:
        parts.append("text_only")
    if args.tabular_only:
        parts.append("tabular_only")
    if args.freeze_text_encoder:
        parts.append("frozen_text")
    return "__".join(parts)


@torch.inference_mode()
def evaluate_loader(model: MadfuseModel, loader: DataLoader, device: str):
    model.eval()
    preds = []
    labels = []
    start = time.perf_counter()
    for batch in loader:
        batch = move_batch(batch, device)
        out = model.forward_single(
            batch["input_ids"],
            batch["attention_mask"],
            batch["cat_ids"],
            batch["num_scaled"],
            batch["num_obs"],
            batch["mask_features"],
        )
        preds.append(out["logits"].argmax(dim=1).detach().cpu().numpy())
        labels.append(batch["labels"].detach().cpu().numpy())
    predict_s = time.perf_counter() - start
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    return compute_classification_metrics(y_true, y_pred), predict_s


def fit_one_seed(args, seed: int, val_experiments: List[str]) -> List[Dict]:
    set_seed(seed)
    train_csv, clean_val_csv = infer_prompt_paths(
        prompt_dir=args.prompt_dir,
        data_prefix=args.data_prefix,
        tier=args.tier,
        train_experiment=args.train_experiment,
        val_experiment=args.train_experiment,
    )
    train_df = pd.read_csv(train_csv)
    clean_val_df = pd.read_csv(clean_val_csv)
    validate_frame(train_df, args.text_col, train_csv)
    validate_frame(clean_val_df, args.text_col, clean_val_csv)

    cat_vocab = fit_category_vocab(train_df)
    num_stats = fit_numeric_stats(train_df)
    if AutoTokenizer is None:
        raise RuntimeError("transformers is required to run MADFUSE-IDS.")
    tokenizer = AutoTokenizer.from_pretrained(rp(args.model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    tokenizer.padding_side = "right"

    train_records = build_records(train_df, args.text_col, cat_vocab, num_stats)
    clean_val_records = build_records(clean_val_df, args.text_col, cat_vocab, num_stats)
    train_loader = DataLoader(
        RecordDataset(train_records),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=TrainCollator(
            tokenizer=tokenizer,
            max_len=args.max_len,
            augment_cfg=AugmentConfig(
                mask_ratios=[float(x) for x in parse_csv_list(args.mask_ratios)],
                mask_modes=parse_csv_list(args.mask_modes),
            ),
        ),
    )
    clean_val_loader = DataLoader(
        RecordDataset(clean_val_records),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=EvalCollator(tokenizer=tokenizer, max_len=args.max_len),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MadfuseModel(args.model_path, cat_vocab, args).to(device)
    params = count_torch_parameters(model)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    method_name = model_variant_name(args)
    run_dir = rp(os.path.join(args.output_root, f"{args.tier}_{method_name}_seed{seed}"))
    ensure_dir(run_dir)

    best_state = None
    best_macro = -1.0
    no_improve = 0
    reset_peak_memory()
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "cls": 0.0, "cons": 0.0, "rec": 0.0, "batches": 0}
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            clean_out = model.forward_single(
                batch["clean_input_ids"],
                batch["clean_attention_mask"],
                batch["cat_ids"],
                batch["clean_num_scaled"],
                batch["clean_num_obs"],
                batch["clean_mask_features"],
            )
            aug_out = model.forward_single(
                batch["aug_input_ids"],
                batch["aug_attention_mask"],
                batch["cat_ids"],
                batch["aug_num_scaled"],
                batch["aug_num_obs"],
                batch["aug_mask_features"],
            )
            loss_cls = 0.5 * (
                F.cross_entropy(clean_out["logits"], batch["labels"])
                + F.cross_entropy(aug_out["logits"], batch["labels"])
            )
            loss_cons = (
                torch.zeros((), device=device)
                if args.no_consistency_loss
                else js_divergence(clean_out["logits"], aug_out["logits"])
            )
            loss_rec = (
                torch.zeros((), device=device)
                if args.no_reconstruction_loss
                else reconstruction_loss(aug_out["rec_logits"], batch["rec_targets"], batch["rec_mask"])
            )
            loss = loss_cls + args.cons_weight * loss_cons + args.rec_weight * loss_rec
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running["loss"] += float(loss.detach().cpu())
            running["cls"] += float(loss_cls.detach().cpu())
            running["cons"] += float(loss_cons.detach().cpu())
            running["rec"] += float(loss_rec.detach().cpu())
            running["batches"] += 1

        clean_metrics, _ = evaluate_loader(model, clean_val_loader, device)
        clean_macro = clean_metrics["macro_f1"]
        denom = max(running["batches"], 1)
        print(
            f"  epoch {epoch:02d} | clean_val_macro_f1={clean_macro:.4f} "
            f"loss={running['loss']/denom:.4f} cls={running['cls']/denom:.4f} "
            f"cons={running['cons']/denom:.4f} rec={running['rec']/denom:.4f}"
        )
        if clean_macro > best_macro + 1e-4:
            best_macro = clean_macro
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break
    train_s = time.perf_counter() - train_start
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    peak_mem = peak_memory_mb()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "cat_vocab": cat_vocab,
            "num_stats": num_stats,
            "args": vars(args),
        },
        os.path.join(run_dir, f"{method_name}__seed-{seed}.pt"),
    )

    rows = []
    for val_experiment in val_experiments:
        _, val_csv = infer_prompt_paths(
            prompt_dir=args.prompt_dir,
            data_prefix=args.data_prefix,
            tier=args.tier,
            train_experiment=args.train_experiment,
            val_experiment=val_experiment,
        )
        val_df = pd.read_csv(val_csv)
        validate_frame(val_df, args.text_col, val_csv)
        val_records = build_records(val_df, args.text_col, cat_vocab, num_stats)
        val_loader = DataLoader(
            RecordDataset(val_records),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=EvalCollator(tokenizer=tokenizer, max_len=args.max_len),
        )
        metrics, predict_s = evaluate_loader(model, val_loader, device)
        runtime = {
            "train_s": float(train_s),
            "predict_s": float(predict_s),
            "inference_latency_ms_per_sample": mean_latency_ms(predict_s, len(val_df)),
            "peak_gpu_mem_mb": float(peak_mem),
        }
        row = flatten_metric_record(
            method_family="madfuse",
            method_name=method_name,
            tier=args.tier,
            text_col=f"{args.text_col}+structured",
            train_experiment=args.train_experiment,
            val_experiment=val_experiment,
            seed=seed,
            split_seed=args.split_seed,
            metrics=metrics,
            runtime=runtime,
            params=params,
            extra={"dataset_name": args.dataset_name},
        )
        payload = {
            **row,
            "dataset_name": args.dataset_name,
            "metrics": metrics,
            "runtime": runtime,
            "params": params,
            "train_csv": train_csv,
            "val_csv": val_csv,
            "model_path": rp(args.model_path),
            "losses": {
                "cons_weight": args.cons_weight,
                "rec_weight": args.rec_weight,
                "no_consistency_loss": bool(args.no_consistency_loss),
                "no_reconstruction_loss": bool(args.no_reconstruction_loss),
            },
            "augment": {
                "mask_ratios": [float(x) for x in parse_csv_list(args.mask_ratios)],
                "mask_modes": parse_csv_list(args.mask_modes),
                "branch_dropout_p": args.branch_dropout_p,
            },
            "experiment_meta": parse_experiment_metadata(val_experiment),
        }
        metrics_path = os.path.join(
            run_dir,
            f"metrics__{method_name.lower()}__seed-{seed}__{val_experiment}.json",
        )
        save_json(metrics_path, payload)
        row["metrics_json"] = os.path.basename(metrics_path)
        rows.append(row)
    return rows


def merge_summary(rows: List[Dict], summary_path: str, train_experiment: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if os.path.exists(summary_path):
        old_df = pd.read_csv(summary_path)
        old_df = old_df.drop(columns=["macro_f1_base", "delta_macro_f1"], errors="ignore")
        df = pd.concat([old_df, df], ignore_index=True)
        key_cols = [
            "method_family",
            "method_name",
            "tier",
            "text_col",
            "train_experiment",
            "val_experiment",
            "seed",
            "split_seed",
        ]
        df = df.drop_duplicates(subset=key_cols, keep="last")
    base = (
        df[df["val_experiment"] == train_experiment][
            ["method_family", "method_name", "text_col", "seed", "split_seed", "macro_f1"]
        ]
        .rename(columns={"macro_f1": "macro_f1_base"})
    )
    df = df.merge(
        base,
        on=["method_family", "method_name", "text_col", "seed", "split_seed"],
        how="left",
    )
    df["delta_macro_f1"] = df["macro_f1"] - df["macro_f1_base"]
    df = df.sort_values(by=["method_name", "seed", "val_experiment"])
    df.to_csv(summary_path, index=False)
    return df


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.text_only and args.tabular_only:
        raise ValueError("text-only and tabular-only cannot both be enabled.")
    ensure_dir(rp(args.output_root))
    val_experiments = parse_csv_list(args.val_experiments)
    seeds = parse_int_list(args.seeds)

    all_rows = []
    for seed in seeds:
        print(f"[RUN] family=madfuse method={model_variant_name(args)} seed={seed} tier={args.tier}")
        seed_rows = fit_one_seed(args, seed, val_experiments)
        all_rows.extend(seed_rows)

    summary_path = os.path.join(rp(args.output_root), f"summary__{args.tier}.csv")
    df = merge_summary(all_rows, summary_path, args.train_experiment)
    print(f"[SUMMARY] saved -> {summary_path} rows={len(df)}")


if __name__ == "__main__":
    main()
