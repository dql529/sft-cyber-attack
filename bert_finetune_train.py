# bert_finetune_train.py —— 无 argparse 版
import os, sys, re, numpy as np, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.utils.class_weight import compute_class_weight

# ---------- 基本配置 ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config import (
    PROMPT_TRAIN_CSV,
    PROMPT_TRAIN_CSV_MINI,
    PROMPT_VAL_CSV,
    PROMPT_VAL_CSV_MINI,
    LABEL_MAP,
)


def rp(p):
    return (
        os.path.join(ROOT, p.lstrip("./"))
        if isinstance(p, str) and (p.startswith("./") or p.startswith("../"))
        else p
    )


USE_TRAIN_MINI = True
USE_VAL_MINI = True
ENCODER = "bert-base-uncased"
OUTPUT_DIR = "./bert_outputs/seqclf"
EPOCHS = 3
TRAIN_BS = 8
EVAL_BS = 16
LR = 2e-5
MAX_LEN = 256
FP16 = True
# -----------------------------

train_csv = rp(PROMPT_TRAIN_CSV_MINI if USE_TRAIN_MINI else PROMPT_TRAIN_CSV)
val_csv = rp(PROMPT_VAL_CSV_MINI if USE_VAL_MINI else PROMPT_VAL_CSV)
os.makedirs(rp(OUTPUT_DIR), exist_ok=True)

num_labels = len(LABEL_MAP)
id2label = {k: v for k, v in LABEL_MAP.items()}
label2id = {v: k for k, v in LABEL_MAP.items()}

print(f"[CFG] ENCODER={ENCODER}  TRAIN={train_csv}  VAL={val_csv}")

TASK_RE = re.compile(
    r"\[Task\].*?Input:\s*(.*?)(?:\n\s*Answer:|\nAnswer:|\n\s*---|\Z)", re.S
)


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = TASK_RE.search(s)
    t = m.group(1).strip() if m else s
    return " ".join(t.split())


# 加载与清洗
train_ds = load_dataset("csv", data_files=train_csv)["train"]
val_ds = load_dataset("csv", data_files=val_csv)["train"]
train_ds = train_ds.map(lambda e: {"text": clean_text(e["text"])})
val_ds = val_ds.map(lambda e: {"text": clean_text(e["text"])})

# Tokenize
tok = AutoTokenizer.from_pretrained(ENCODER)


def preprocess(ex):
    return tok(ex["text"], truncation=True, max_length=MAX_LEN)


train_tok = train_ds.map(preprocess, batched=True)
val_tok = val_ds.map(preprocess, batched=True)

train_tok = train_tok.rename_column("label", "labels")
val_tok = val_tok.rename_column("label", "labels")
cols = ["input_ids", "attention_mask", "labels"]
train_tok.set_format(type="torch", columns=cols)
val_tok.set_format(type="torch", columns=cols)

# 类别权重
y = np.array(train_tok["labels"], dtype=int)
classes = np.arange(num_labels)
cw = compute_class_weight("balanced", classes=classes, y=y)
cw = torch.tensor(cw, dtype=torch.float)

# 模型
model = AutoModelForSequenceClassification.from_pretrained(
    ENCODER, num_labels=num_labels, id2label=id2label, label2id=label2id
)


# 加权 loss 的 Trainer
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"].to(model.device)
        outputs = model(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
        )
        logits = outputs.logits
        loss = torch.nn.CrossEntropyLoss(weight=cw.to(model.device))(
            logits.view(-1, model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


args_tr = TrainingArguments(
    output_dir=rp(OUTPUT_DIR),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=FP16,
    logging_steps=100,
)


def metrics_fn(p):
    from sklearn.metrics import accuracy_score, f1_score

    preds = p.predictions.argmax(1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1_macro": f1_score(p.label_ids, preds, average="macro"),
    }


trainer = WeightedTrainer(
    model=model,
    args=args_tr,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    tokenizer=tok,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=metrics_fn,
)
trainer.train()
save_dir = rp(os.path.join(OUTPUT_DIR, "best_model"))
trainer.save_model(save_dir)
tok.save_pretrained(save_dir)
print("Saved best model to:", save_dir)
