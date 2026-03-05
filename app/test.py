# test.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

model_path = "base_models/qwen2.5-7b-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("\n=== 设备检查 ===\n")

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

print("\n加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16 if device == "cuda" else torch.float32
)
model.to(device)
model.eval()

print("\n=== 模型就绪 ===.\n")

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
        streamer=streamer
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
