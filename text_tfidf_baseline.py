"""
Same-text non-LLM baselines for the robustness study.

This isolates the gain from frozen pretrained representations versus plain
bag-of-ngrams on the exact same prompt text.
"""

import argparse
import os
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from robustness_utils import (
    DEFAULT_VAL_EXPERIMENTS,
    compute_classification_metrics,
    ensure_dir,
    estimate_linear_probe_params,
    flatten_metric_record,
    infer_prompt_paths,
    mean_latency_ms,
    parse_csv_list,
    parse_int_list,
    rp,
    save_json,
    set_seed,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-dir", default="./data/prompt_csv")
    ap.add_argument("--data-prefix", default="unsw15_prompt")
    ap.add_argument("--tier", choices=["mini", "medium", "full"], default="medium")
    ap.add_argument("--train-experiment", default="E1_clean")
    ap.add_argument("--val-experiments", default=",".join(DEFAULT_VAL_EXPERIMENTS))
    ap.add_argument("--train-csv-override", default="")
    ap.add_argument("--val-csv-override", default="")
    ap.add_argument("--dataset-label-override", default="")
    ap.add_argument("--text-cols", default="text")
    ap.add_argument("--models", default="tfidf_logreg,tfidf_svm")
    ap.add_argument("--seeds", default="42,52,62")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--max-features", type=int, default=50000)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--output-root", default="./text_baseline_outputs")
    return ap.parse_args()


def build_estimator(name: str, seed: int):
    if name == "tfidf_logreg":
        return LogisticRegression(
            max_iter=2000,
            solver="saga",
            class_weight="balanced",
            random_state=seed,
        )
    if name == "tfidf_svm":
        return LinearSVC(class_weight="balanced", random_state=seed)
    raise ValueError(f"Unknown model '{name}'.")


def load_text_frame(csv_path: str, text_col: str):
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise RuntimeError(f"{csv_path} is missing text column '{text_col}'.")
    if "label" not in df.columns:
        raise RuntimeError(f"{csv_path} is missing column 'label'.")
    return df[text_col].astype(str).tolist(), df["label"].astype(int).to_numpy(), df


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
    label_names = [label_name_map.get(i, f"class_{i}") for i in label_ids]
    return label_ids, label_names


def main():
    args = parse_args()
    ensure_dir(rp(args.output_root))
    val_experiments = parse_csv_list(args.val_experiments)
    text_cols = parse_csv_list(args.text_cols)
    models = parse_csv_list(args.models)
    seeds = parse_int_list(args.seeds)

    rows = []
    for seed in seeds:
        set_seed(seed)
        for text_col in text_cols:
            train_csv, clean_val_csv = infer_prompt_paths(
                prompt_dir=args.prompt_dir,
                data_prefix=args.data_prefix,
                tier=args.tier,
                train_experiment=args.train_experiment,
                val_experiment=args.train_experiment,
                train_override=args.train_csv_override,
                val_override=args.val_csv_override or args.train_csv_override,
            )
            try:
                train_texts, y_tr, train_df = load_text_frame(train_csv, text_col)
                _, _, clean_val_df = load_text_frame(clean_val_csv, text_col)
            except RuntimeError as exc:
                print(f"[SKIP] {exc}")
                continue
            label_ids, label_names = infer_label_metadata(train_df, clean_val_df)

            vectorizer = TfidfVectorizer(
                max_features=args.max_features,
                ngram_range=(1, args.ngram_max),
                lowercase=True,
            )
            vector_start = time.perf_counter()
            X_tr = vectorizer.fit_transform(train_texts)
            vectorize_train_s = time.perf_counter() - vector_start

            for model_name in models:
                estimator = build_estimator(model_name, seed)
                train_start = time.perf_counter()
                estimator.fit(X_tr, y_tr)
                train_s = time.perf_counter() - train_start
                params = {
                    "total_params": estimate_linear_probe_params(estimator),
                    "trainable_params": estimate_linear_probe_params(estimator),
                    "vocab_size": len(vectorizer.vocabulary_),
                }

                for val_experiment in val_experiments:
                    _, val_csv = infer_prompt_paths(
                        prompt_dir=args.prompt_dir,
                        data_prefix=args.data_prefix,
                        tier=args.tier,
                        train_experiment=args.train_experiment,
                        val_experiment=val_experiment,
                        train_override=args.train_csv_override,
                        val_override=args.val_csv_override,
                    )
                    try:
                        val_texts, y_true, val_df = load_text_frame(val_csv, text_col)
                    except RuntimeError as exc:
                        print(f"[SKIP] {exc}")
                        continue
                    run_label_ids, run_label_names = infer_label_metadata(train_df, val_df)

                    vec_val_start = time.perf_counter()
                    X_va = vectorizer.transform(val_texts)
                    vectorize_val_s = time.perf_counter() - vec_val_start
                    pred_start = time.perf_counter()
                    y_pred = estimator.predict(X_va)
                    predict_s = time.perf_counter() - pred_start
                    metrics = compute_classification_metrics(
                        y_true,
                        y_pred,
                        label_ids=run_label_ids or label_ids,
                        label_names=run_label_names or label_names,
                    )
                    runtime = {
                        "vectorize_train_s": float(vectorize_train_s),
                        "vectorize_val_s": float(vectorize_val_s),
                        "train_s": float(train_s),
                        "predict_s": float(predict_s),
                        "inference_latency_ms_per_sample": mean_latency_ms(predict_s, len(y_true)),
                        "peak_gpu_mem_mb": 0.0,
                    }
                    row = flatten_metric_record(
                        method_family="text_baseline",
                        method_name=model_name,
                        tier=args.tier,
                        text_col=text_col,
                        train_experiment=args.train_experiment,
                        val_experiment=val_experiment,
                        seed=seed,
                        split_seed=args.split_seed,
                        metrics=metrics,
                        runtime=runtime,
                        params=params,
                        extra={
                            "dataset_label": args.dataset_label_override or val_experiment,
                        },
                    )
                    payload = {
                        **row,
                        "train_csv": train_csv,
                        "val_csv": val_csv,
                        "metrics": metrics,
                        "runtime": runtime,
                        "params": params,
                    }
                    out_dir = rp(args.output_root)
                    metrics_path = os.path.join(
                        out_dir,
                        (
                            f"metrics__{model_name}__"
                            f"dataset-{(args.dataset_label_override or val_experiment).replace('/', '_')}__"
                            f"col-{text_col}__seed-{seed}__{val_experiment}.json"
                        ),
                    )
                    save_json(metrics_path, payload)
                    row["metrics_json"] = os.path.basename(metrics_path)
                    rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
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
        base = (
            df[df["val_experiment"] == args.train_experiment][
                ["method_name", "text_col", "dataset_label", "seed", "split_seed", "macro_f1"]
            ].rename(columns={"macro_f1": "macro_f1_base"})
        )
        df = df.merge(
            base,
            on=["method_name", "text_col", "dataset_label", "seed", "split_seed"],
            how="left",
        )
        df["delta_macro_f1"] = df["macro_f1"] - df["macro_f1_base"]

    summary_path = os.path.join(rp(args.output_root), f"summary__{args.tier}.csv")
    df.to_csv(summary_path, index=False)
    print(f"[SUMMARY] saved -> {summary_path}")


if __name__ == "__main__":
    main()
