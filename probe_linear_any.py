"""
Frozen-encoder linear probes for the robustness study.

This script supports:
- fixed clean-train / degraded-val evaluation;
- prompt-format ablations (`text` vs `text_nl`);
- multiple frozen backbones, including a smaller Gemma checkpoint;
- multi-seed probe runs on a fixed split;
- rich artifact saving for reporting and later aggregation.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import dump
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

from robustness_utils import (
    DEFAULT_VAL_EXPERIMENTS,
    LABEL_IDS,
    LABEL_MAP,
    compute_classification_metrics,
    ensure_dir,
    estimate_linear_probe_params,
    experiment_label,
    flatten_metric_record,
    infer_prompt_paths,
    mean_latency_ms,
    parse_csv_list,
    parse_int_list,
    peak_memory_mb,
    reset_peak_memory,
    rp,
    save_json,
    set_seed,
)


DEFAULT_MODELS = ["bert", "gemma3-270m", "deepseek13"]
PROBE_STORE: Dict[str, object] = {}
PARAM_CACHE: Dict[str, Optional[int]] = {}

PRESET_MODELS: Dict[str, Dict] = {
    "bert": {
        "path": "./models/bert-base-uncased",
        "default_max_len": 256,
        "default_batch": 16,
        "trust_remote_code": False,
    },
    "gemma3-270m": {
        "path": "./models/gemma-3-270m",
        "default_max_len": 256,
        "default_batch": 16,
        "trust_remote_code": True,
    },
    "deepseek13": {
        "path": "./models/deepseek-coder-1.3b-base",
        "default_max_len": 128,
        "default_batch": 4,
        "trust_remote_code": True,
    },
    "deepseek67": {
        "path": "./models/deepseek-coder-6.7b-base",
        "default_max_len": 128,
        "default_batch": 1,
        "trust_remote_code": True,
    },
    "minilm-sentence": {
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "default_max_len": 256,
        "default_batch": 32,
        "trust_remote_code": False,
    },
}

_TASK_ONLY_RE = re.compile(r"\[Task\]\s*Input:\s*(.*?)\s*Answer\s*:", flags=re.S | re.I)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-dir", default="./data/prompt_csv")
    ap.add_argument("--data-prefix", default="unsw15_prompt")
    ap.add_argument("--tier", choices=["mini", "medium", "full"], default="medium")
    ap.add_argument("--train-experiment", default="E1_clean")
    ap.add_argument(
        "--val-experiments",
        default=",".join(DEFAULT_VAL_EXPERIMENTS),
        help="Comma-separated validation experiments.",
    )
    ap.add_argument("--train-csv-override", default="")
    ap.add_argument("--val-csv-override", default="")
    ap.add_argument("--dataset-label-override", default="")
    ap.add_argument("--text-cols", default="text,text_nl")
    ap.add_argument("--cleaning", default="none", choices=["none", "task_only", "raw_prompt"])
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--head", default="logreg", choices=["logreg", "sgd"])
    ap.add_argument("--l2norm", action="store_true")
    ap.add_argument("--truncation-side", default="auto", choices=["auto", "left", "right"])
    ap.add_argument("--max-len-override", type=int, default=0)
    ap.add_argument("--batch-override", type=int, default=0)
    ap.add_argument("--seeds", default="42,52,62")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--output-root", default="./probe_outputs_any")
    return ap.parse_args()


def apply_cleaning(texts: List[str], mode: str) -> List[str]:
    mode = str(mode).lower()
    if mode in {"none", "no", "off"}:
        return texts
    if mode == "raw_prompt":
        return texts
    if mode == "task_only":
        out = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            match = _TASK_ONLY_RE.search(text)
            out.append(match.group(1).strip() if match else text.strip())
        return out
    raise ValueError(f"Unknown cleaning mode: {mode}")


def safe_basename(path: str) -> str:
    base = os.path.basename(str(path).rstrip("/\\"))
    return base.replace("/", "_").replace("\\", "_").replace(":", "_")


def pick_dtype():
    return torch.float16 if torch.cuda.is_available() else None


def l2_normalize(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(n, eps, None)


def load_backbone(model_path: str, trust_remote_code: bool):
    if AutoTokenizer is None or AutoModel is None:
        raise RuntimeError("transformers is required to run frozen-probe experiments.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype()

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.unk_token
    tok.padding_side = "right"

    need_hidden_states = False
    try:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except Exception:
        causal = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        base = (
            getattr(causal, "model", None)
            or getattr(causal, "transformer", None)
            or getattr(causal, "base_model", None)
        )
        if base is not None:
            model = base
        else:
            model = causal
            need_hidden_states = True

    model.to(device)
    model.eval()
    return tok, model, device, need_hidden_states


@torch.inference_mode()
def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    need_hidden_states: bool,
    max_len: int,
    batch_size: int,
    truncation_side: str,
) -> np.ndarray:
    tokenizer.truncation_side = truncation_side
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        if need_hidden_states:
            out = model(**enc, use_cache=False, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[-1]
        else:
            out = model(**enc, return_dict=True)
            hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vectors.append(pooled.detach().cpu().numpy())
    return np.vstack(vectors)


def get_or_fit_probe(
    probe_id: str,
    head: str,
    seed: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
) -> Tuple[object, float]:
    if probe_id in PROBE_STORE:
        return PROBE_STORE[probe_id], 0.0

    if head == "sgd":
        clf = make_pipeline(
            StandardScaler(with_mean=True),
            SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                max_iter=2000,
                tol=1e-3,
                random_state=seed,
                class_weight="balanced",
            ),
        )
    else:
            clf = make_pipeline(
                StandardScaler(with_mean=True),
                LogisticRegression(
                    max_iter=2000,
                    solver="saga",
                    tol=1e-3,
                    random_state=seed,
                    class_weight="balanced",
                ),
            )
    fit_start = time.perf_counter()
    clf.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - fit_start
    PROBE_STORE[probe_id] = clf
    return clf, fit_s


def load_csv_bundle(csv_path: str, text_col: str, cleaning: str) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise RuntimeError(f"{csv_path} is missing text column '{text_col}'.")
    if "label" not in df.columns:
        raise RuntimeError(f"{csv_path} is missing column 'label'.")
    texts = apply_cleaning(df[text_col].astype(str).tolist(), cleaning)
    labels = df["label"].astype(int).to_numpy()
    return texts, labels, df


def infer_label_metadata(*frames: pd.DataFrame):
    label_ids = sorted(
        {
            int(v)
            for df in frames
            if "label" in df.columns
            for v in df["label"].dropna().astype(int).tolist()
        }
    )
    label_name_map = {}
    for df in frames:
        if {"label", "label_name"}.issubset(df.columns):
            pairs = df[["label", "label_name"]].dropna().drop_duplicates()
            for label, name in pairs.itertuples(index=False):
                label_name_map[int(label)] = str(name)
    if not label_ids:
        label_ids = list(LABEL_IDS)
    label_names = [label_name_map.get(i, LABEL_MAP.get(i, f"class_{i}")) for i in label_ids]
    return label_ids, label_names


def cache_feature_paths(
    out_dir: str,
    train_csv: str,
    val_csv: str,
    model_tag: str,
    model_path: str,
    text_col: str,
    max_len: int,
    cleaning: str,
    truncation_side: str,
) -> Tuple[str, str]:
    cache_dir = os.path.join(out_dir, "cache")
    ensure_dir(cache_dir)
    train_key = (
        f"{safe_basename(train_csv)}__{model_tag}__{safe_basename(model_path)}"
        f"__col-{text_col}__L{max_len}__{cleaning}__trunc-{truncation_side}"
    )
    val_key = (
        f"{safe_basename(val_csv)}__{model_tag}__{safe_basename(model_path)}"
        f"__col-{text_col}__L{max_len}__{cleaning}__trunc-{truncation_side}"
    )
    return (
        os.path.join(cache_dir, f"X_train__{train_key}.npy"),
        os.path.join(cache_dir, f"X_val__{val_key}.npy"),
    )


def run_one_probe(
    *,
    model_tag: str,
    model_cfg: Dict,
    train_csv: str,
    val_csv: str,
    out_dir: str,
    dataset_label: str,
    text_col: str,
    cleaning: str,
    head: str,
    l2norm: bool,
    truncation_side: str,
    max_len_override: int,
    batch_override: int,
    seed: int,
    split_seed: int,
    tier: str,
    train_experiment: str,
    val_experiment: str,
) -> Dict:
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Val CSV not found: {val_csv}")

    max_len = int(max_len_override or model_cfg["default_max_len"])
    batch_size = int(batch_override or model_cfg["default_batch"])
    model_path = model_cfg["path"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    trunc = "left" if truncation_side == "auto" and cleaning == "raw_prompt" else truncation_side
    if trunc == "auto":
        trunc = "right"

    train_texts, y_tr, train_df = load_csv_bundle(train_csv, text_col, cleaning)
    val_texts, y_va, val_df = load_csv_bundle(val_csv, text_col, cleaning)
    label_ids, label_names = infer_label_metadata(train_df, val_df)
    cache_tr, cache_va = cache_feature_paths(
        out_dir, train_csv, val_csv, model_tag, model_path, text_col, max_len, cleaning, trunc
    )

    reset_peak_memory()
    encode_train_s = 0.0
    encode_val_s = 0.0
    cache_train_load_s = 0.0
    cache_val_load_s = 0.0
    backbone_total_params = PARAM_CACHE.get(model_path)

    tok = model = device = need_hidden_states = None

    if os.path.exists(cache_tr):
        t0 = time.perf_counter()
        X_tr = np.load(cache_tr)
        cache_train_load_s = time.perf_counter() - t0
    else:
        tok, model, device, need_hidden_states = load_backbone(model_path, trust_remote_code)
        backbone_total_params = sum(p.numel() for p in model.parameters())
        PARAM_CACHE[model_path] = int(backbone_total_params)
        t0 = time.perf_counter()
        X_tr = encode_texts(
            train_texts, tok, model, device, need_hidden_states, max_len, batch_size, trunc
        )
        encode_train_s = time.perf_counter() - t0
        np.save(cache_tr, X_tr.astype(np.float16))

    if os.path.exists(cache_va):
        t0 = time.perf_counter()
        X_va = np.load(cache_va)
        cache_val_load_s = time.perf_counter() - t0
    else:
        if tok is None:
            tok, model, device, need_hidden_states = load_backbone(model_path, trust_remote_code)
            backbone_total_params = sum(p.numel() for p in model.parameters())
            PARAM_CACHE[model_path] = int(backbone_total_params)
        t0 = time.perf_counter()
        X_va = encode_texts(
            val_texts, tok, model, device, need_hidden_states, max_len, batch_size, trunc
        )
        encode_val_s = time.perf_counter() - t0
        np.save(cache_va, X_va.astype(np.float16))

    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    X_tr = X_tr.astype(np.float32, copy=False)
    X_va = X_va.astype(np.float32, copy=False)
    if l2norm:
        X_tr = l2_normalize(X_tr)
        X_va = l2_normalize(X_va)

    probe_id = (
        f"{safe_basename(train_csv)}__{model_tag}__col-{text_col}__seed-{seed}"
        f"__head-{head}__L{max_len}__{cleaning}__trunc-{trunc}__l2-{int(l2norm)}"
    )
    clf, probe_train_s = get_or_fit_probe(probe_id, head, seed, X_tr, y_tr)

    pred_start = time.perf_counter()
    y_pred = clf.predict(X_va)
    predict_s = time.perf_counter() - pred_start

    metrics = compute_classification_metrics(
        y_va,
        y_pred,
        label_ids=label_ids,
        label_names=label_names,
    )
    probe_params = estimate_linear_probe_params(clf)
    total_params = None if backbone_total_params is None else int(backbone_total_params) + int(probe_params)
    params = {
        "backbone_total_params": None if backbone_total_params is None else int(backbone_total_params),
        "backbone_trainable_params": 0,
        "probe_trainable_params": int(probe_params),
        "total_params": total_params,
        "trainable_params": int(probe_params),
    }
    runtime = {
        "encode_train_s": float(encode_train_s),
        "encode_val_s": float(encode_val_s),
        "cache_train_load_s": float(cache_train_load_s),
        "cache_val_load_s": float(cache_val_load_s),
        "probe_train_s": float(probe_train_s),
        "predict_s": float(predict_s),
        "inference_latency_ms_per_sample": mean_latency_ms(predict_s, len(y_va)),
        "peak_gpu_mem_mb": peak_memory_mb(),
    }

    dataset_tag = safe_basename(dataset_label)
    key = (
        f"{dataset_tag}__{model_tag}__col-{text_col}__seed-{seed}"
        f"__head-{head}__l2-{int(l2norm)}"
    )
    ensure_dir(out_dir)
    joblib_path = os.path.join(out_dir, f"linear_probe__{key}.joblib")
    metrics_path = os.path.join(out_dir, f"metrics__{key}.json")
    meta = {
        "method_family": "frozen_probe",
        "method_name": model_tag,
        "dataset": dataset_label,
        "model": model_tag,
        "model_path": model_path,
        "tier": tier,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "text_col": text_col,
        "cleaning": cleaning,
        "truncation_side": trunc,
        "head": head,
        "l2norm": bool(l2norm),
        "max_len": max_len,
        "batch": batch_size,
        "seed": int(seed),
        "split_seed": int(split_seed),
        "train_experiment": train_experiment,
        "val_experiment": val_experiment,
        "metrics": metrics,
        "runtime": runtime,
        "params": params,
    }
    dump({"pipe": clf, "meta": meta}, joblib_path)
    save_json(metrics_path, meta)

    return flatten_metric_record(
        method_family="frozen_probe",
        method_name=model_tag,
        tier=tier,
        text_col=text_col,
        train_experiment=train_experiment,
        val_experiment=val_experiment,
        seed=seed,
        split_seed=split_seed,
        metrics=metrics,
        runtime=runtime,
        params=params,
        extra={
            "dataset_label": dataset_label,
            "joblib": os.path.basename(joblib_path),
            "metrics_json": os.path.basename(metrics_path),
            "cleaning": cleaning,
            "head": head,
            "l2norm": int(l2norm),
        },
    )


def run_suite(args) -> pd.DataFrame:
    models = parse_csv_list(args.models)
    text_cols = parse_csv_list(args.text_cols)
    val_experiments = parse_csv_list(args.val_experiments)
    seeds = parse_int_list(args.seeds)

    for model_name in models:
        if model_name not in PRESET_MODELS:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(PRESET_MODELS)}")

    rows = []
    for seed in seeds:
        set_seed(seed)
        for text_col in text_cols:
            for val_experiment in val_experiments:
                train_csv, val_csv = infer_prompt_paths(
                    prompt_dir=args.prompt_dir,
                    data_prefix=args.data_prefix,
                    tier=args.tier,
                    train_experiment=args.train_experiment,
                    val_experiment=val_experiment,
                    train_override=args.train_csv_override,
                    val_override=args.val_csv_override,
                )
                ds_label = args.dataset_label_override or experiment_label(
                    args.train_experiment, val_experiment
                )
                out_dir = os.path.join(rp(args.output_root), f"{args.tier}_{ds_label}")
                for model_tag in models:
                    print(
                        f"[RUN] family=frozen_probe model={model_tag} text_col={text_col} "
                        f"seed={seed} dataset={ds_label}"
                    )
                    try:
                        row = run_one_probe(
                            model_tag=model_tag,
                            model_cfg=PRESET_MODELS[model_tag],
                            train_csv=train_csv,
                            val_csv=val_csv,
                            out_dir=out_dir,
                            dataset_label=ds_label,
                            text_col=text_col,
                            cleaning=args.cleaning,
                            head=args.head,
                            l2norm=args.l2norm,
                            truncation_side=args.truncation_side,
                            max_len_override=args.max_len_override,
                            batch_override=args.batch_override,
                            seed=seed,
                            split_seed=args.split_seed,
                            tier=args.tier,
                            train_experiment=args.train_experiment,
                            val_experiment=val_experiment,
                        )
                    except RuntimeError as exc:
                        if "missing text column" in str(exc):
                            print(f"[SKIP] {exc}")
                            continue
                        raise
                    rows.append(row)

    df = pd.DataFrame(rows)
    ensure_dir(rp(args.output_root))
    summary_path = os.path.join(rp(args.output_root), f"summary__{args.tier}.csv")
    if os.path.exists(summary_path):
        old_df = pd.read_csv(summary_path)
        old_df = old_df.drop(columns=["macro_f1_base", "delta_macro_f1"], errors="ignore")
        df = pd.concat([old_df, df], ignore_index=True)
        key_cols = [
            "method_family",
            "method_name",
            "tier",
            "text_col",
            "dataset_label",
            "train_experiment",
            "val_experiment",
            "seed",
            "split_seed",
        ]
        df = df.drop_duplicates(subset=key_cols, keep="last")
    if not df.empty:
        base = (
            df[df["val_experiment"] == args.train_experiment][
                [
                    "method_family",
                    "method_name",
                    "text_col",
                    "dataset_label",
                    "seed",
                    "split_seed",
                    "macro_f1",
                ]
            ]
            .rename(columns={"macro_f1": "macro_f1_base"})
        )
        df = df.merge(
            base,
            on=[
                "method_family",
                "method_name",
                "text_col",
                "dataset_label",
                "seed",
                "split_seed",
            ],
            how="left",
        )
        df["delta_macro_f1"] = df["macro_f1"] - df["macro_f1_base"]
    df.sort_values(
        by=["text_col", "method_name", "seed", "val_experiment"],
        ascending=[True, True, True, True],
    ).to_csv(summary_path, index=False)
    print(f"[SUMMARY] saved -> {summary_path}")
    return df


def main():
    args = parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
