# clear_cache_mini.py
import os, glob

PATTERNS = [
    r"./bert_outputs/X_train__mini__bert-base-uncased__L256__task__mean__tsR__unsw15_prompt_train_mini__unsw15_prompt_val_mini.npy",
    r"./bert_outputs/X_val__mini__bert-base-uncased__L256__task__mean__tsR__unsw15_prompt_train_mini__unsw15_prompt_val_mini.npy",
    r"./bert_outputs/linear_probe__mini__*.joblib",  # 旧头也清理掉免干扰
]

for pat in PATTERNS:
    for p in glob.glob(pat):
        try:
            os.remove(p)
            print("deleted:", p)
        except Exception as e:
            print("failed:", p, e)
