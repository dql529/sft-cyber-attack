import os

# 关键修复：在导入torch之前，通过环境变量彻底禁用Inductor和Dynamo
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import re
import random
import pandas as pd
import torch
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import classification_report
from datasets import Dataset

# --- 路径配置 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config import *


def resolve_path(path_str):
    if isinstance(path_str, str) and (
        path_str.startswith("./") or path_str.startswith("../")
    ):
        return os.path.join(project_root, path_str)
    return path_str


# 解析所有路径
BASE_MODEL_PATH = resolve_path(BASE_MODEL_PATH)
PROMPT_TRAIN_CSV = resolve_path(PROMPT_TRAIN_CSV)
PROMPT_VAL_CSV = resolve_path(PROMPT_VAL_CSV)
OUTPUT_DIR = resolve_path(OUTPUT_DIR)
RESULTS_DIR = resolve_path("./results/")
# --- 路径配置结束 ---

# --- 全局变量 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = os.path.join(RESULTS_DIR, "evaluation_results.csv")
# ---


def save_report_to_csv(report_dict, model_name, scenario):
    records = []
    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict):
            records.append(
                {
                    "model": model_name,
                    "scenario": scenario,
                    "class": class_name,
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1-score": metrics.get("f1-score"),
                    "support": metrics.get("support"),
                }
            )
    df = pd.DataFrame(records)
    if not os.path.exists(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, index=False)
    else:
        df.to_csv(RESULTS_FILE, mode="a", header=False, index=False)
    print(f"✅ Results for '{scenario}' saved to {RESULTS_FILE}")


def pretty_print_report(report_dict):
    """使用pandas美化打印评估报告"""
    df = pd.DataFrame(report_dict).transpose()
    print(df[["precision", "recall", "f1-score", "support"]].to_string())


def extract_label(text, all_labels):
    for label in all_labels:
        if re.search(rf'["\'`\s]{label}["\'`\s,.]', f" {text} "):
            return label
    for label in all_labels:
        if label in text:
            return label
    return ""


def create_few_shot_prompt(test_prompt, examples, num_shots):
    if num_shots == 0:
        return test_prompt
    shot_examples = random.sample(examples, num_shots)
    few_shot_prompt = ""
    for ex in shot_examples:
        label_name = LABEL_MAP[ex[1]]
        few_shot_prompt += ex[0] + " " + label_name + "\n\n---\n\n"
    return few_shot_prompt + test_prompt


def evaluate_model(
    model, tokenizer, val_file, label_map, few_shot_examples=None, num_shots=0
):
    df = pd.read_csv(val_file, nrows=100)
    y_true = []
    y_pred = []
    model.eval()
    label2id = {v: k for k, v in label_map.items()}

    for i, row in tqdm(
        df.iterrows(), total=df.shape[0], desc=f"Evaluating ({num_shots}-shot)"
    ):
        test_prompt = row["text"]
        prompt = create_few_shot_prompt(test_prompt, few_shot_examples, num_shots)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )
        predicted_label_str = extract_label(generated_text, list(label_map.values()))

        if predicted_label_str in label2id:
            y_pred.append(label2id[predicted_label_str])
        else:
            y_pred.append(-1)
        y_true.append(row["label"])

    return classification_report(
        y_true,
        y_pred,
        labels=list(label_map.keys()),
        target_names=list(label_map.values()),
        zero_division=0,
        output_dict=True,
    )


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
        print(f"🗑️ Cleared old results file: {RESULTS_FILE}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True, attn_implementation="eager"
    ).to(device)

    model_name_for_report = os.path.basename(BASE_MODEL_PATH)

    print("=" * 30)
    print("1. 开始评估 Baseline 模型 (Zero-Shot)...")
    print("=" * 30)
    report_dict_zero_shot = evaluate_model(
        base_model, tokenizer, PROMPT_VAL_CSV, LABEL_MAP, num_shots=0
    )
    print("\nZero-Shot (Baseline) 模型评估报告:\n")
    pretty_print_report(report_dict_zero_shot)
    save_report_to_csv(report_dict_zero_shot, model_name_for_report, "Zero-Shot")

    print("\n" + "=" * 30)
    print("2. 开始评估 Baseline 模型 (Few-Shot)...")
    print("=" * 30)

    train_df = pd.read_csv(PROMPT_TRAIN_CSV)
    few_shot_examples = list(zip(train_df["text"], train_df["label"]))

    for n_shots in [1, 5]:
        print(f"\n--- Evaluating with {n_shots}-shot ---")
        report_dict_few_shot = evaluate_model(
            base_model,
            tokenizer,
            PROMPT_VAL_CSV,
            LABEL_MAP,
            few_shot_examples=few_shot_examples,
            num_shots=n_shots,
        )
        print(f"\n{n_shots}-Shot 模型评估报告:\n")
        pretty_print_report(report_dict_few_shot)
        save_report_to_csv(
            report_dict_few_shot, model_name_for_report, f"{n_shots}-Shot"
        )

    print("\n" + "=" * 30)
    print("3. 开始评估 Fine-Tuned 模型 (LoRA)...")
    print("=" * 30)

    if not os.path.exists(os.path.join(OUTPUT_DIR, "adapter_config.json")):
        print(f"INFO: 在 '{OUTPUT_DIR}' 中找不到微调模型，将跳过此步骤。")
    else:
        finetuned_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR).to(device)
        report_dict_finetuned = evaluate_model(
            finetuned_model, tokenizer, PROMPT_VAL_CSV, LABEL_MAP
        )
        print("\nFine-Tuned 模型评估报告:\n")
        pretty_print_report(report_dict_finetuned)
        save_report_to_csv(report_dict_finetuned, model_name_for_report, "Fine-Tuned")

    print("\n" + "=" * 30)
    print(f"🎉 所有评估完成！结果已保存至 {RESULTS_FILE}")
    print("=" * 30)
