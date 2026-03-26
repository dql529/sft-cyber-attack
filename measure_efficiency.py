from __future__ import annotations

import argparse
import gc
import ctypes
import json
import os
import pickle
import sys
import threading
import time
from pathlib import Path
from ctypes import wintypes

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FILTERED_DATA_CSV
from nn_tabular_baseline import (
    BATCH_SIZE,
    MLP,
    NumpyDataset,
    build_augmented_train_df,
    count_torch_parameters,
    load_data,
    train_torch_model,
)
from robustness_utils import ensure_dir, estimate_linear_probe_params, mean_latency_ms, set_seed


PROCESS_HANDLE = ctypes.windll.kernel32.GetCurrentProcess()
PSAPI = ctypes.WinDLL("psapi")


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


PSAPI.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
    wintypes.DWORD,
]
PSAPI.GetProcessMemoryInfo.restype = wintypes.BOOL


def current_rss_bytes() -> int:
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    ok = PSAPI.GetProcessMemoryInfo(PROCESS_HANDLE, ctypes.byref(counters), counters.cb)
    if not ok:
        return 0
    return int(counters.WorkingSetSize)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Measure lightweight efficiency metrics on the aligned UNSW split.")
    ap.add_argument("--tier", choices=["mini", "medium", "full"], default="medium")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--model-seed", type=int, default=11)
    ap.add_argument("--dataset-csv", default=FILTERED_DATA_CSV)
    ap.add_argument("--feature-mode", default="available")
    ap.add_argument("--split-index", default="./data/prompt_csv/splits_unsw15_seed{seed}.npz")
    ap.add_argument("--fair-text-dir", default="./data/fair_text_csv/seed42")
    ap.add_argument("--fair-text-prefix", default="unsw_structured_text")
    ap.add_argument("--maskaug-label", default="MLP_MaskAug")
    ap.add_argument("--train-mask-ratios", default="0.1,0.3")
    ap.add_argument("--train-block-ratios", default="0.1")
    ap.add_argument("--train-noise-ratios", default="")
    ap.add_argument("--output-dir", default="./runs/reviewer_shield/efficiency")
    return ap.parse_args()


def parse_ratio_list(value: str):
    items = []
    for raw in str(value).split(","):
        token = raw.strip()
        if not token:
            continue
        items.append(float(token))
    return items


def sample_peak_rss_mb(func):
    stop = threading.Event()
    peak = {"rss": current_rss_bytes()}

    def sampler():
        while not stop.is_set():
            rss = current_rss_bytes()
            if rss > peak["rss"]:
                peak["rss"] = rss
            time.sleep(0.002)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    stop.set()
    thread.join(timeout=0.05)
    return result, elapsed, peak["rss"] / (1024 ** 2)


def predict_torch_cpu(model: torch.nn.Module, X: np.ndarray):
    model = model.to("cpu").eval()
    ds = NumpyDataset(X.astype(np.float32), np.zeros(len(X), dtype=np.int64))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def build_text_estimator(model_name: str):
    if model_name == "tfidf_logreg":
        return LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="saga",
            class_weight="balanced",
            random_state=42,
        )
    if model_name == "tfidf_svm":
        return LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    raise ValueError(f"Unsupported text model: {model_name}")


def benchmark_text(train_csv: Path, val_csv: Path, output_dir: Path, model_name: str) -> dict:
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    train_texts = train_df["text"].astype(str).tolist()
    y_tr = train_df["label"].astype(int).to_numpy()
    val_texts = val_df["text"].astype(str).tolist()

    train_start = time.perf_counter()
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        lowercase=True,
        min_df=1,
        max_df=1.0,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )
    X_tr = vectorizer.fit_transform(train_texts)
    clf = build_text_estimator(model_name)
    clf.fit(X_tr, y_tr)
    train_s = time.perf_counter() - train_start

    model_path = output_dir / f"{model_name}_pipeline.joblib"
    joblib.dump({"vectorizer": vectorizer, "model": clf}, model_path)

    def infer():
        X_val = vectorizer.transform(val_texts)
        return clf.predict(X_val)

    _, infer_s, peak_rss_mb = sample_peak_rss_mb(infer)
    n_samples = len(val_df)
    throughput = float(n_samples / infer_s) if infer_s > 0 else 0.0
    return {
        "method_name": "TF-IDF + Logistic Regression" if model_name == "tfidf_logreg" else "TF-IDF + SVM",
        "model_size_mb": float(model_path.stat().st_size / (1024 ** 2)),
        "training_time_s": float(train_s),
        "cpu_inference_latency_ms": mean_latency_ms(infer_s, n_samples),
        "cpu_throughput_flows_s": throughput,
        "peak_cpu_ram_mb": float(peak_rss_mb),
        "peak_gpu_mem_mb": np.nan,
        "trainable_params": int(estimate_linear_probe_params(clf)),
    }


def benchmark_tabular(args: argparse.Namespace, model_name: str, use_augmented: bool, output_dir: Path) -> dict:
    set_seed(args.model_seed)
    total_start = time.perf_counter()
    train_df, val_df, transformer, _, num_cols, feature_cols, X_tr, y_tr, label_ids, label_names = load_data(
        tier=args.tier,
        split_seed=args.split_seed,
        split_index=args.split_index,
        dataset_csv=args.dataset_csv,
        feature_mode=args.feature_mode,
    )
    if use_augmented:
        train_aug_df = build_augmented_train_df(
            train_df,
            num_cols,
            args.split_seed,
            parse_ratio_list(args.train_mask_ratios),
            parse_ratio_list(args.train_block_ratios),
            parse_ratio_list(args.train_noise_ratios),
        )
        X_fit = transformer.transform(train_aug_df[feature_cols]).astype(np.float32)
        y_fit = train_aug_df["label"].to_numpy()
    else:
        X_fit = X_tr
        y_fit = y_tr

    X_val = transformer.transform(val_df[feature_cols]).astype(np.float32)
    y_val = val_df["label"].to_numpy()
    model = MLP(X_fit.shape[1], len(label_ids))
    model, _, peak_gpu_mem_mb = train_torch_model(model, X_fit, y_fit, X_val, y_val, label_ids, label_names)
    total_train_s = time.perf_counter() - total_start

    state_path = output_dir / f"{model_name.replace(' ', '_').lower()}_seed{args.model_seed}.pt"
    torch.save(model.state_dict(), state_path)
    model = model.to("cpu")
    gc.collect()

    def infer():
        return predict_torch_cpu(model, X_val)

    _, infer_s, peak_rss_mb = sample_peak_rss_mb(infer)
    n_samples = len(y_val)
    throughput = float(n_samples / infer_s) if infer_s > 0 else 0.0
    params = count_torch_parameters(model)
    return {
        "method_name": model_name,
        "model_size_mb": float(state_path.stat().st_size / (1024 ** 2)),
        "training_time_s": float(total_train_s),
        "cpu_inference_latency_ms": mean_latency_ms(infer_s, n_samples),
        "cpu_throughput_flows_s": throughput,
        "peak_cpu_ram_mb": float(peak_rss_mb),
        "peak_gpu_mem_mb": float(peak_gpu_mem_mb) if torch.cuda.is_available() else np.nan,
        "trainable_params": int(params["trainable_params"]),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(str(output_dir))

    fair_dir = Path(args.fair_text_dir).resolve()
    train_csv = fair_dir / f"{args.fair_text_prefix}_train_{args.tier}_E1_clean.csv"
    val_csv = fair_dir / f"{args.fair_text_prefix}_val_{args.tier}_E1_clean.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Missing fair text bundle: {train_csv} | {val_csv}")

    rows = [
        benchmark_text(train_csv, val_csv, output_dir, "tfidf_logreg"),
        benchmark_text(train_csv, val_csv, output_dir, "tfidf_svm"),
        benchmark_tabular(args, "MLP", use_augmented=False, output_dir=output_dir),
        benchmark_tabular(args, args.maskaug_label, use_augmented=True, output_dir=output_dir),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"efficiency_table__{args.tier}.csv", index=False)

    meta = {
        "cpu": "12th Gen Intel(R) Core(TM) i9-12900KS",
        "gpu": "NVIDIA GeForce RTX 3090 Ti",
        "system_ram_gb_approx": 32,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "fair_text_train_csv": str(train_csv),
        "fair_text_val_csv": str(val_csv),
    }
    with open(output_dir / "efficiency_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[EFFICIENCY] saved -> {output_dir}")


if __name__ == "__main__":
    main()
