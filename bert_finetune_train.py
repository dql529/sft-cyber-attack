# bert_finetune_train.py
import os, sys, argparse, numpy as np, torch, re
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.utils.class_weight import compute_class_weight

# --- resolve project root & config ---
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
        if p.startswith("./") or p.startswith("../")
        else p
    )


# --- args ---
ap = argparse.ArgumentParser()
ap.add_argument("--encoder", default="bert-base-uncased")
ap.add_argument("--use_train_mini", action="store_true")
ap.add_argument("--use_val_mini", action="store_true")
ap.add_argument("--output_dir", default="./bert_outputs/seqclf")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--train_bs", type=int, default=16)
ap.add_argument("--eval_bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--fp16", action="store_true")
args = ap.parse_args()

train_csv = rp(PROMPT_TRAIN_CSV_MINI if args.use_train_mini else PROMPT_TRAIN_CSV)
val_csv = rp(PROMPT_VAL_CSV_MINI if args.use_val_mini else PROMPT_VAL_CSV)
num_labels = len(LABEL_MAP)
id2label = {k: v for k, v in LABEL_MAP.items()}
label2id = {v: k for k, v in LABEL_MAP.items()}

print("Encoder:", args.encoder)
print("Train CSV:", train_csv)
print("Val   CSV:", val_csv)

# --- prompt cleaning: 只取 [Task] Input: ... 到 Answer: 之前 ---
TASK_RE = re.compile(
    r"\[Task\].*?Input:\s*(.*?)(?:\n\s*Answer:|\nAnswer:|\n\s*---|\Z)", re.S
)


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = TASK_RE.search(s)
    text = m.group(1).strip() if m else s
    return " ".join(text.split())


# --- load datasets ---
train_ds = load_dataset("csv", data_files=train_csv)["train"]
val_ds = load_dataset("csv", data_files=val_csv)["train"]

# apply cleaning
train_ds = train_ds.map(lambda e: {"text": clean_text(e["text"])})
val_ds = val_ds.map(lambda e: {"text": clean_text(e["text"])})

# tokenizer & preprocess
tok = AutoTokenizer.from_pretrained(args.encoder)


def preprocess(ex):
    return tok(ex["text"], truncation=True, max_length=args.max_len)


train_tok = train_ds.map(preprocess, batched=True)
val_tok = val_ds.map(preprocess, batched=True)

train_tok = train_tok.rename_column("label", "labels")
val_tok = val_tok.rename_column("label", "labels")
cols = ["input_ids", "attention_mask", "labels"]
train_tok.set_format(type="torch", columns=cols)
val_tok.set_format(type="torch", columns=cols)

# class weights
y = np.array(train_tok["labels"], dtype=int)
classes = np.arange(num_labels)
cw = compute_class_weight("balanced", classes=classes, y=y)
cw = torch.tensor(cw, dtype=torch.float)

# model
model = AutoModelForSequenceClassification.from_pretrained(
    args.encoder, num_labels=num_labels, id2label=id2label, label2id=label2id
)


# weighted loss
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
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=args.train_bs,
    per_device_eval_batch_size=args.eval_bs,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=args.fp16,
    logging_steps=100,
)


def metrics_fn(p):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    preds = np.argmax(p.predictions, axis=1)
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
save_dir = os.path.join(args.output_dir, "best_model")
trainer.save_model(save_dir)
tok.save_pretrained(save_dir)
print("Saved best model to:", save_dir)
