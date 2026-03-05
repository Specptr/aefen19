# test.py
# 26.3.5
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig
)
from threading import Thread
from config import *

model_path = BASE_MODEL

print("\n=== 设备检查 ===\n")

if torch.cuda.is_available():
    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("Running on CPU")

print("\n加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("加载模型...")

if device == "cuda":

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

else:

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

    model.to(device)

model.eval()

print("\n=== 模型就绪 ===\n")

history = []

while True:

    user = input("你: ").strip()

    if user.lower() in ["exit", "quit"]:
        break

    history.append({"role": "user", "content": user})

    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=200,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        streamer=streamer,
        repetition_penalty=1.1
    )

    print("模型: ", end="", flush=True)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    response = ""

    for token in streamer:
        print(token, end="", flush=True)
        response += token

    print("\n")

    history.append({"role": "assistant", "content": response})
