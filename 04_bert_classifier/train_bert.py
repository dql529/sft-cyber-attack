import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )
except ImportError:
    Dataset = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    Trainer = None
    TrainingArguments = None
    default_data_collator = None

# Disable compile backends to keep training stable in this environment.
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from robustness_utils import (
    DEFAULT_VAL_EXPERIMENTS,
    compute_classification_metrics,
    count_torch_parameters,
    ensure_dir,
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
from config import LABEL_MAP


MODEL_NAME = "./models/bert-base-uncased"
NUM_LABELS = len(LABEL_MAP)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-dir", default="./data/prompt_csv")
    ap.add_argument("--data-prefix", default="unsw15_prompt")
    ap.add_argument("--model-path", default=MODEL_NAME)
    ap.add_argument("--tier", choices=["mini", "medium", "full"], default="medium")
    ap.add_argument("--train-experiment", default="E1_clean")
    ap.add_argument("--val-experiments", default=",".join(DEFAULT_VAL_EXPERIMENTS))
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--seeds", default="42,52,62")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--train-batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--output-root", default="./checkpoints/bert_classifier_robustness")
    return ap.parse_args()


def tokenize_frame(df: pd.DataFrame, tokenizer, text_col: str, max_len: int):
    if Dataset is None:
        raise RuntimeError("datasets is required to run the fine-tuned BERT comparator.")
    if text_col not in df.columns:
        raise RuntimeError(f"CSV is missing text column '{text_col}'.")
    if "label" not in df.columns:
        raise RuntimeError("CSV is missing column 'label'.")
    dataset = Dataset.from_pandas(df[[text_col, "label"]].rename(columns={text_col: "text"}))

    def tokenize_function(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_len)

    return dataset.map(tokenize_function, batched=True)


def trainer_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def training_args_for(run_dir: str, args, seed: int):
    if TrainingArguments is None:
        raise RuntimeError("transformers is required to run the fine-tuned BERT comparator.")
    common = dict(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        logging_dir=os.path.join(run_dir, "logs"),
        logging_steps=100,
        save_strategy="no",
        report_to=[],
        seed=seed,
        data_seed=seed,
        disable_tqdm=False,
    )
    try:
        return TrainingArguments(**common, evaluation_strategy="epoch")
    except TypeError:
        return TrainingArguments(**common)


def run_one_seed(args, seed: int, val_experiments: list[str]):
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or Trainer is None:
        raise RuntimeError(
            "transformers and datasets are required to run the fine-tuned BERT comparator."
        )
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

    model_path = rp(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=NUM_LABELS,
        local_files_only=True,
    )
    params = count_torch_parameters(model)
    run_dir = rp(os.path.join(args.output_root, f"{args.tier}_{args.text_col}_seed{seed}"))
    ensure_dir(run_dir)

    tokenized_train = tokenize_frame(train_df, tokenizer, args.text_col, args.max_len)
    tokenized_clean_val = tokenize_frame(clean_val_df, tokenizer, args.text_col, args.max_len)

    trainer = Trainer(
        model=model,
        args=training_args_for(run_dir, args, seed),
        train_dataset=tokenized_train,
        eval_dataset=tokenized_clean_val,
        compute_metrics=trainer_metrics,
        data_collator=default_data_collator,
    )

    reset_peak_memory()
    train_start = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - train_start
    peak_mem = peak_memory_mb()
    trainer.save_model(run_dir)

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
        tokenized_val = tokenize_frame(val_df, tokenizer, args.text_col, args.max_len)
        predict_start = time.perf_counter()
        pred_out = trainer.predict(tokenized_val)
        predict_s = time.perf_counter() - predict_start
        y_true = val_df["label"].astype(int).to_numpy()
        y_pred = np.argmax(pred_out.predictions, axis=1)
        metrics = compute_classification_metrics(y_true, y_pred)
        runtime = {
            "train_s": float(train_s),
            "predict_s": float(predict_s),
            "inference_latency_ms_per_sample": mean_latency_ms(predict_s, len(y_true)),
            "peak_gpu_mem_mb": float(peak_mem),
        }
        row = flatten_metric_record(
            method_family="finetune",
            method_name="bert_finetune",
            tier=args.tier,
            text_col=args.text_col,
            train_experiment=args.train_experiment,
            val_experiment=val_experiment,
            seed=seed,
            split_seed=args.split_seed,
            metrics=metrics,
            runtime=runtime,
            params=params,
        )
        payload = {
            **row,
            "dataset_label": experiment_label(args.train_experiment, val_experiment),
            "model_name": model_path,
            "train_csv": train_csv,
            "val_csv": val_csv,
            "metrics": metrics,
            "runtime": runtime,
            "params": params,
            "checkpoint_dir": run_dir,
        }
        metrics_path = os.path.join(
            run_dir, f"metrics__bert_finetune__seed-{seed}__{val_experiment}.json"
        )
        save_json(metrics_path, payload)
        row["metrics_json"] = os.path.basename(metrics_path)
        rows.append(row)
    return rows


def main():
    args = parse_args()
    ensure_dir(rp(args.output_root))
    val_experiments = parse_csv_list(args.val_experiments)
    seeds = parse_int_list(args.seeds)

    rows = []
    for seed in seeds:
        print(f"[RUN] finetune model=bert seed={seed} text_col={args.text_col}")
        rows.extend(run_one_seed(args, seed, val_experiments))

    df = pd.DataFrame(rows)
    summary_path = os.path.join(rp(args.output_root), f"summary__{args.tier}_{args.text_col}.csv")
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
    if not df.empty:
        base = (
            df[df["val_experiment"] == args.train_experiment][["seed", "split_seed", "macro_f1"]]
            .rename(columns={"macro_f1": "macro_f1_base"})
        )
        df = df.merge(base, on=["seed", "split_seed"], how="left")
        df["delta_macro_f1"] = df["macro_f1"] - df["macro_f1_base"]

    df.to_csv(summary_path, index=False)
    print(f"[SUMMARY] saved -> {summary_path}")


if __name__ == "__main__":
    main()
