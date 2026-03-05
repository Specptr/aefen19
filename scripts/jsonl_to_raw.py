# jsonl_to_raw.py
# 26.3.4
import json
from pathlib import Path
from config import *

input_file = PROCESSED_DATA
output_file = RAW_DATA

def convert():
    dialogue_count = 0
    sentence_count = 0

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        for line in f_in:
            item = json.loads(line)

            messages = item.get("messages", [])
            dialogue_count += 1

            for msg in messages:
                role = msg["role"]
                content = msg["content"].strip()

                f_out.write(f"{role}: {content}\n")
                sentence_count += 1

            f_out.write("\n")

    print("\n=== Conversion Finished ===")
    print(f"Dialogues: {dialogue_count}")
    print(f"Sentences: {sentence_count}")
    print(f"Output file: {output_file}\n")

if __name__ == "__main__":
    convert()
