import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import LABEL_MAP


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VAL_EXPERIMENTS = [
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
LABEL_IDS = sorted(LABEL_MAP.keys())
LABEL_NAMES = [LABEL_MAP[i] for i in LABEL_IDS]


def rp(path: str) -> str:
    if isinstance(path, str) and (path.startswith("./") or path.startswith("../")):
        return os.path.join(ROOT, path.lstrip("./"))
    return path


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_csv_list(value: str | Sequence[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def parse_int_list(value: str | Sequence[int] | None) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def tier_tag(tier: str) -> str:
    tier = str(tier).lower()
    if tier == "full":
        return ""
    if tier in {"mini", "medium"}:
        return f"{tier}_"
    raise ValueError(f"Unknown tier '{tier}'.")


def infer_prompt_paths(
    prompt_dir: str,
    data_prefix: str,
    tier: str,
    train_experiment: str,
    val_experiment: str,
    train_override: str = "",
    val_override: str = "",
) -> Tuple[str, str]:
    prompt_dir = rp(prompt_dir)
    tag = tier_tag(tier)
    if train_override:
        train_csv = os.path.abspath(rp(train_override))
    else:
        train_csv = os.path.abspath(
            os.path.join(prompt_dir, f"{data_prefix}_train_{tag}{train_experiment}.csv")
        )
    if val_override:
        val_csv = os.path.abspath(rp(val_override))
    else:
        val_csv = os.path.abspath(
            os.path.join(prompt_dir, f"{data_prefix}_val_{tag}{val_experiment}.csv")
        )
    return train_csv, val_csv


def experiment_label(train_experiment: str, val_experiment: str) -> str:
    if train_experiment == val_experiment:
        return val_experiment
    return f"train-{train_experiment}__val-{val_experiment}"


def parse_experiment_metadata(experiment: str) -> Dict[str, float | str | int]:
    if experiment == "E1_clean":
        return {"family": "clean", "rate": 0, "label": "clean"}
    if experiment.startswith("E3_mask_"):
        rate = int(experiment.split("_")[-1])
        return {"family": "mask", "rate": rate, "label": f"mask_{rate}"}
    if experiment.startswith("E3_block_"):
        rate = int(experiment.split("_")[-1])
        return {"family": "block", "rate": rate, "label": f"block_{rate}"}
    if experiment.startswith("E4_noise_"):
        rate = int(experiment.split("_")[-1])
        return {"family": "noise", "rate": rate, "label": f"noise_{rate}"}
    return {"family": "other", "rate": -1, "label": str(experiment)}


def normalize_label_schema(
    label_ids: Sequence[int] | None = None,
    label_names: Sequence[str] | Mapping[int, str] | None = None,
):
    ids = list(label_ids) if label_ids is not None else list(LABEL_IDS)
    if label_names is None:
        names = [LABEL_MAP.get(i, f"class_{i}") for i in ids]
    elif isinstance(label_names, Mapping):
        names = [str(label_names.get(i, f"class_{i}")) for i in ids]
    else:
        names = [str(x) for x in label_names]
        if len(names) != len(ids):
            raise ValueError("label_names length must match label_ids length.")
    return ids, names


def compute_classification_metrics(y_true, y_pred, label_ids=None, label_names=None) -> Dict:
    label_ids, label_names = normalize_label_schema(label_ids, label_names)
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
    weighted = f1_score(
        y_true, y_pred, labels=label_ids, average="weighted", zero_division=0
    )
    per_class = f1_score(
        y_true, y_pred, labels=label_ids, average=None, zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    return {
        "acc": float(acc),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "per_class_f1": {label_names[i]: float(v) for i, v in enumerate(per_class)},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "label_ids": [int(x) for x in label_ids],
        "label_names": [str(x) for x in label_names],
    }


def save_json(path: str, payload: Dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def count_torch_parameters(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def estimate_linear_probe_params(pipe) -> int:
    clf = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe
    total = 0
    if hasattr(clf, "coef_"):
        total += int(np.size(clf.coef_))
    if hasattr(clf, "intercept_"):
        total += int(np.size(clf.intercept_))
    return total


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def mean_latency_ms(elapsed_s: float, n_samples: int) -> float:
    if n_samples <= 0:
        return 0.0
    return float(elapsed_s * 1000.0 / n_samples)


def flatten_metric_record(
    method_family: str,
    method_name: str,
    tier: str,
    text_col: str,
    train_experiment: str,
    val_experiment: str,
    seed: int,
    split_seed: int,
    metrics: Dict,
    runtime: Dict | None = None,
    params: Dict | None = None,
    extra: Dict | None = None,
) -> Dict:
    row = {
        "method_family": method_family,
        "method_name": method_name,
        "tier": tier,
        "text_col": text_col,
        "train_experiment": train_experiment,
        "val_experiment": val_experiment,
        "seed": int(seed),
        "split_seed": int(split_seed),
        "acc": metrics["acc"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }
    if runtime:
        row.update(runtime)
    if params:
        row.update(params)
    if extra:
        row.update(extra)
    return row


METHOD_NAME_MAP = {
    "MLP_MaskAug": "MLP MaskAug",
    "MLP": "MLP",
    "tfidf_svm": "TF-IDF + SVM",
    "tfidf_logreg": "TF-IDF + Logistic Regression",
    "Logistic Regression": "Structured Logistic Regression",
    "Linear SVM": "Structured Linear SVM",
    "MLP_MaskOnly": "MLP MaskOnly",
    "MLP_BlockOnly": "MLP BlockOnly",
}

CONDITION_NAME_MAP = {
    "E1_clean": "Clean",
    "E3_mask_10": "Mask10",
    "E3_mask_30": "Mask30",
    "E3_mask_50": "Mask50",
    "E3_block_10": "Block10",
    "E3_block_30": "Block30",
    "E3_block_50": "Block50",
    "E4_noise_10": "Noise10",
    "E4_noise_30": "Noise30",
    "E4_noise_50": "Noise50",
}


def canonical_method_name(method_name: str) -> str:
    return METHOD_NAME_MAP.get(str(method_name), str(method_name))


def canonical_condition_name(val_experiment: str) -> str:
    return CONDITION_NAME_MAP.get(str(val_experiment), str(val_experiment))


def slugify_token(value: str) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("-", "_")
        .replace("/", "_")
    )


def flatten_per_class_report(
    metrics: Dict,
    dataset: str,
    method_name: str,
    condition: str,
    seed: int | None = None,
    split_seed: int | None = None,
) -> List[Dict]:
    report = metrics.get("classification_report", {})
    label_names = metrics.get("label_names", [])
    rows = []
    for label_name in label_names:
        cls = report.get(label_name, {})
        if not cls:
            continue
        rows.append(
            {
                "dataset": str(dataset),
                "method": canonical_method_name(method_name),
                "condition": canonical_condition_name(condition),
                "seed": int(seed) if seed is not None else None,
                "split_seed": int(split_seed) if split_seed is not None else None,
                "class": str(label_name),
                "precision": float(cls.get("precision", 0.0)),
                "recall": float(cls.get("recall", 0.0)),
                "f1": float(cls.get("f1-score", 0.0)),
                "support": int(round(float(cls.get("support", 0.0)))),
            }
        )
    return rows
