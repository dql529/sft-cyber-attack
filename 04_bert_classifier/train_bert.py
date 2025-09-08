import os

# 关键修复：在导入torch之前，通过环境变量彻底禁用Inductor和Dynamo
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import sys
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# --- 路径配置 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config import *

def resolve_path(path_str):
    if isinstance(path_str, str) and (path_str.startswith('./') or path_str.startswith('../')):
        return os.path.join(project_root, path_str)
    return path_str

PROMPT_TRAIN_CSV = resolve_path(PROMPT_TRAIN_CSV)
PROMPT_VAL_CSV = resolve_path(PROMPT_VAL_CSV)
# --- 路径配置结束 ---

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = len(LABEL_MAP)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

if __name__ == "__main__":
    print(f"--- Starting BERT Fine-Tuning for Text Classification ---")
    print(f"Model: {MODEL_NAME}")

    # 加载Tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    # 加载数据
    train_df = pd.read_csv(PROMPT_TRAIN_CSV)
    val_df = pd.read_csv(PROMPT_VAL_CSV)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Tokenize数据
    print("🔄 Tokenizing data...")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./checkpoints/bert_classifier",
        num_train_epochs=3,
        per_device_train_batch_size=32, # BERT模型较小，可以使用更大的批大小
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    # 开始训练
    print("🚀 Starting training...")
    trainer.train()

    # 评估并打印最终结果
    print("\n--- Final Evaluation ---")
    eval_results = trainer.evaluate()
    print(f"\n✅ Final Evaluation Results:\n{eval_results}")
