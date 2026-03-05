# raw_to_jsonl.py
# 26.3.4
import json
from config import *

INPUT_FILE = RAW_DATA
OUTPUT_FILE = PROCESSED_DATA

def parse_dialogues(lines):
    dialogues = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        if line.startswith("user:"):
            user = line[len("user:"):].strip()

            if i + 1 < n and lines[i + 1].strip().startswith("assistant:"):
                assistant = lines[i + 1].strip()[len("assistant:"):].strip()

                if user and assistant:
                    dialogues.append((user, assistant))

                i += 2
                continue

        i += 1

    return dialogues

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

pairs = parse_dialogues(lines)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for user, assistant in pairs:
        obj = {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ]
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("\n=== Conversion Finished ===")
print(f"Converted {len(pairs)} dialogues")
print(f"Output file: {OUTPUT_FILE}\n")
