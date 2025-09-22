# Shared configurations for the project

# --- Paths ---
# BASE_MODEL_PATH = "./models/gemma-3-270m"
BASE_MODEL_PATH = "./models/deepseek-coder-1.3b-base"
# BASE_MODEL_PATH = "./models/bert-base-uncased"

RAW_DATA_CSV = "./data/Augmented-UNSW_NB15.csv"
FILTERED_DATA_CSV = "./data/unsw15_filtered_nolog.csv"

PROMPT_TRAIN_CSV = "./data/prompt_csv/unsw15_prompt_train.csv"
PROMPT_TRAIN_CSV_MINI = "./data/prompt_csv/unsw15_prompt_train_mini.csv"

PROMPT_VAL_CSV = "./data/prompt_csv/unsw15_prompt_val.csv"
PROMPT_VAL_CSV_MINI = "./data/prompt_csv/unsw15_prompt_val_mini.csv"

TOKENIZED_TRAIN_PATH = "./cached/tokenized_train"
TOKENIZED_VAL_PATH = "./cached/tokenized_val"
# config.py
PROMPT_TRAIN_CSV_MEDIUM = "./data/prompt_csv/unsw15_prompt_train_medium.csv"
PROMPT_VAL_CSV_MEDIUM = "./data/prompt_csv/unsw15_prompt_val_medium.csv"

OUTPUT_DIR = "./checkpoints/lora_gemma_clm"

# --- Labels ---
LABEL_MAP = {
    0: "Backdoor",
    1: "DoS",
    2: "Exploits",
    3: "Normal",
    4: "Reconnaissance",
    5: "Shellcode",
    6: "Worms",
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

# --- Lora Config ---
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_DROPOUT = 0.1

# --- Training Args ---
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 24
GRAD_ACCUMULATION_STEPS = 2
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-4
