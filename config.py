# =====================
# model
# =====================

BASE_MODEL = "base_models/qwen2.5-7b-instruct"
ADAPTER_DIR = "adapters/..."

USE_4BIT = True

# =====================
# data
# =====================

RAW_DATA = "data/raw/.txt"

PROCESSED_DATA = "data/processed/.jsonl"

TOKENIZED_DATA = "data/tokenized/..."

# =====================
# training
# =====================

MAX_LENGTH = 2048

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

TRAIN_BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 3
LR = 5e-5

# =====================
# inference
# =====================

GEN_MAX_NEW_TOKENS = 256
TEMPERATURE = 0.85
TOP_P = 0.85

# =====================
# persona
# =====================

PHOTO_PATH = "assets/poetic.png"

APP_TITLE = "Æfen 19"

SYSTEM_MSG = {
    "role": "system",
    "content": (
        "你的名字是...。\n"
    )
}
