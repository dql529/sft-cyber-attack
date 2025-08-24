# predict_baseline.py
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report
import re
from config import BASE_MODEL_PATH, PROMPT_VAL_CSV, LABEL_MAP, ID2LABEL

# ✅ 路径配置
model_name = BASE_MODEL_PATH
val_file = PROMPT_VAL_CSV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(
    device
)
model.eval()

# ✅ 加载验证集数据
df = pd.read_csv(val_file)


# ✅ 推理函数：生成并提取预测类别
def extract_label(text):
    match = re.search(r"Answer:\s*(\w+)", text)
    return match.group(1) if match else ""


y_true = []
y_pred = []

print("🔍 正在进行 zero-shot 推理...")

for _, row in df.iterrows():
    prompt = row["text"]  # 含 Answer:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_label = extract_label(full_output).strip()

    if predicted_label in ID2LABEL:
        y_pred.append(ID2LABEL[predicted_label])
        y_true.append(row["label"])
    else:
        # 未识别类别时标记为错误（可设为 -1）
        y_pred.append(-1)
        y_true.append(row["label"])

# ✅ 评估
print("\n📊 分类评估报告（未微调 baseline）")
print(classification_report(y_true, y_pred, target_names=list(LABEL_MAP.values())))
