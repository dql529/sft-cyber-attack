# -*- coding: utf-8 -*-
# ========= Windows/CUDA 稳定开关 =========
# finetune
import os

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import sys, shutil, json, hashlib, random
import pandas as pd
from datasets import Dataset, load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

# ========= 路径 & 配置 =========
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config import *  # 复用你的常量


def resolve_path(p):
    if isinstance(p, str) and (p.startswith("./") or p.startswith("../")):
        return os.path.join(project_root, p)
    return p


BASE_MODEL_PATH = resolve_path(BASE_MODEL_PATH)
PROMPT_TRAIN_CSV = resolve_path(PROMPT_TRAIN_CSV)
PROMPT_VAL_CSV = resolve_path(PROMPT_VAL_CSV)
TOKENIZED_TRAIN_PATH = resolve_path(TOKENIZED_TRAIN_PATH)
TOKENIZED_VAL_PATH = resolve_path(TOKENIZED_VAL_PATH)
OUTPUT_DIR = resolve_path(OUTPUT_DIR)

# ========= Tokenizer & 模型 =========
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"  # ★ 保留尾部（含 Answer: <LABEL>）

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, trust_remote_code=True, attn_implementation="eager"
)

# 吞吐优化（可选）
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ========= LoRA 配置 & 注入 =========
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,  # 确保至少包含 q_proj/k_proj/v_proj/o_proj
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# ★ 与 gradient checkpointing 配套的关键三行
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

model.print_trainable_parameters()


# ========= 预处理：只监督“答案片段” =========
# CSV 的 text 已包含 "... Answer:"；我们只训练 " <LABEL>" 的 <LABEL> token
def preprocess_data(examples, max_len=256):
    batch = {k: [] for k in ["input_ids", "attention_mask", "labels"]}
    for text, label_id in zip(examples["text"], examples["label"]):
        y = LABEL_MAP[label_id]  # 例如 "Exploits"
        prompt_part = text  # 含 Answer: 前缀
        answer_part = " " + y

        prompt_ids = tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_part, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids  # 仅答案参与损失

        # 截断到 max_len（保尾部）
        if len(input_ids) > max_len:
            input_ids = input_ids[-max_len:]
            labels = labels[-max_len:]

        # 左侧 PAD；attention_mask: PAD=0, 非PAD=1
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = [tokenizer.pad_token_id] * pad_len + input_ids
            labels = [-100] * pad_len + labels
        attn = [0] * pad_len + [1] * (len(input_ids) - pad_len)

        batch["input_ids"].append(input_ids)
        batch["attention_mask"].append(attn)
        batch["labels"].append(labels)
    return batch


# ========= 缓存签名（自动识别预处理/模板变化） =========
PREPROC_SIGNATURE = {
    "max_len": 256,
    "truncation_side": "left",
    "supervise": "answer_only",
    "collator": "default_data_collator",
    "prompt_version": "v2_class_list_only_one",  # 你改模板时改这个字符串
}


def _sig_hex(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()


PREPROC_SIG_HEX = _sig_hex(PREPROC_SIGNATURE)


def build_or_load_datasets():
    sig_file = os.path.join(
        os.path.dirname(TOKENIZED_TRAIN_PATH), "tokenized_signature.json"
    )

    if (
        os.path.exists(TOKENIZED_TRAIN_PATH)
        and os.path.exists(TOKENIZED_VAL_PATH)
        and os.path.exists(sig_file)
    ):
        try:
            with open(sig_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("sig_hex") == PREPROC_SIG_HEX:
                print("✅ 缓存签名一致，加载 tokenized 数据集...")
                return load_from_disk(TOKENIZED_TRAIN_PATH), load_from_disk(
                    TOKENIZED_VAL_PATH
                )
            else:
                print("⚠️ 缓存签名不一致，将重建 tokenized 数据...")
        except Exception:
            print("⚠️ 读取缓存签名失败，将重建 tokenized 数据...")

    print("🔄 重建 tokenize 数据...")
    for p in [TOKENIZED_TRAIN_PATH, TOKENIZED_VAL_PATH]:
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)

    train_df = pd.read_csv(PROMPT_TRAIN_CSV)
    val_df = pd.read_csv(PROMPT_VAL_CSV)
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    train_ds = train_ds.map(
        lambda ex: preprocess_data(ex, max_len=PREPROC_SIGNATURE["max_len"]),
        batched=True,
        remove_columns=["text", "label"],
    )
    val_ds = val_ds.map(
        lambda ex: preprocess_data(ex, max_len=PREPROC_SIGNATURE["max_len"]),
        batched=True,
        remove_columns=["text", "label"],
    )

    os.makedirs(os.path.dirname(TOKENIZED_TRAIN_PATH), exist_ok=True)
    train_ds.save_to_disk(TOKENIZED_TRAIN_PATH)
    val_ds.save_to_disk(TOKENIZED_VAL_PATH)

    with open(sig_file, "w", encoding="utf-8") as f:
        json.dump(
            {"sig_hex": PREPROC_SIG_HEX, "detail": PREPROC_SIGNATURE},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("💾 已保存新 tokenized 数据及签名。")
    return train_ds, val_ds


# ========= Sanity Checks =========
def debug_tail(ds, idx=0, tail_tokens=60):
    ids = ds[idx]["input_ids"]
    labs = ds[idx]["labels"]
    seq = [t for t in ids if t != tokenizer.pad_token_id]
    tail_txt = tokenizer.decode(seq[-tail_tokens:]) if seq else ""
    tail_labs = labs[-tail_tokens:]
    print("TAIL TEXT:", tail_txt.replace("\n", "\\n"))
    print("TAIL LABEL IDs (last {}):".format(tail_tokens), tail_labs)
    assert any(x != -100 for x in tail_labs), "❌ 尾部没有监督（labels 全是 -100）"


def debug_forward(ds):
    # 临时关闭 checkpointing 做一次梯度连通性检查
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    model.train()
    batch = {k: torch.tensor(v[:2]).to(model.device) for k, v in ds[:2].items()}
    out = model(**batch)
    loss = out.loss
    loss.backward()
    print("forward loss:", float(loss))

    grad_ok = any(
        p.requires_grad and p.grad is not None for _, p in model.named_parameters()
    )
    assert grad_ok, "❌ 未见到可训练参数的梯度——LoRA/预处理可能有问题"

    # 恢复 checkpointing（训练时开启）
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception:
        pass


# ========= 版本兼容的 TrainingArguments 构造 =========
def build_training_args(quick=True):
    lr = max(LEARNING_RATE, 2e-4)  # LoRA 小模型建议 >= 2e-4
    base = dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=lr,
        warmup_ratio=0.1,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
    )
    try:
        if quick:
            print("\nℹ️  NOTE: Running in quick validation mode (max_steps=200).")
            return TrainingArguments(
                **base,
                max_steps=200,
                logging_steps=20,
                save_strategy="no",
                evaluation_strategy="no",
            )
        else:
            print("\nℹ️  NOTE: Full training mode (3 epochs).")
            return TrainingArguments(
                **base,
                num_train_epochs=3,
                weight_decay=0.01,
                logging_steps=100,
                save_strategy="epoch",
                evaluation_strategy="no",
            )
    except TypeError:
        # 兼容非常老的 transformers（无 evaluation/save_strategy 等）
        base.pop("report_to", None)
        if quick:
            return TrainingArguments(**base, max_steps=200, logging_steps=20)
        else:
            return TrainingArguments(**base, num_train_epochs=3, logging_steps=100)


# ========= 训练入口 =========
if __name__ == "__main__":
    random.seed(42)

    train_dataset, val_dataset = build_or_load_datasets()

    # ——— Sanity Check: 10 分钟内能做完 ———
    print("\n[Sanity A] 样本尾部检查（必须含 'Answer: <LABEL>' 且 labels!= -100）")
    debug_tail(train_dataset, idx=0, tail_tokens=60)

    print("\n[Sanity B] 单步前向+反传（loss & 梯度）")
    debug_forward(train_dataset)

    # 快验 / 全量
    QUICK_VALIDATION = True
    args = build_training_args(quick=QUICK_VALIDATION)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,  # 新版会提示用 processing_class，无伤大雅
        data_collator=default_data_collator,  # ★ 关键：不要用 MLM collator
    )

    print("\n" + "=" * 30)
    print("🚀 开始 LoRA 微调训练...")
    print("=" * 30)
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ 训练完成，最终模型已保存：", OUTPUT_DIR)
