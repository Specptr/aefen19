# build_sft_dataset.py
# 26.3.4
import os
import json
from datasets import Dataset
from transformers import AutoTokenizer
from config import *

MODEL_PATH = BASE_MODEL
JSONL_FILE = PROCESSED_DATA
SAVE_PATH = TOKENIZED_DATA

MAX_LENGTH = 2048

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def load_jsonl(path):

    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            msgs = obj.get("messages")

            if len(msgs) >= 2:
                data.append({"messages": msgs})

    return data

raw = load_jsonl(JSONL_FILE)

print("loaded dialogues:", len(raw))

ds = Dataset.from_list(raw)

def tokenize_and_mask(example):

    messages = example["messages"]

    input_ids = []
    labels = []

    for i in range(len(messages)):

        prefix = messages[:i + 1]

        text = tokenizer.apply_chat_template(
            prefix,
            tokenize=False,
            add_generation_prompt=False
        )

        ids = tokenizer(
            text,
            add_special_tokens=False
        )["input_ids"]

        if i == 0:
            continue

        role = messages[i]["role"]

        if role == "assistant":

            prefix_text = tokenizer.apply_chat_template(
                messages[:i],
                tokenize=False,
                add_generation_prompt=True
            )

            prefix_ids = tokenizer(
                prefix_text,
                add_special_tokens=False
            )["input_ids"]

            full_ids = ids[:MAX_LENGTH]

            input_ids = full_ids
            labels = full_ids.copy()

            prefix_len = min(len(prefix_ids), len(full_ids))

            for j in range(prefix_len):
                labels[j] = -100

            break

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels
    }


tokenized = ds.map(
    tokenize_and_mask,
    remove_columns=ds.column_names
)

os.makedirs(SAVE_PATH, exist_ok=True)

tokenized.save_to_disk(SAVE_PATH)

print("saved dataset:", SAVE_PATH)
print("dataset size:", len(tokenized))
