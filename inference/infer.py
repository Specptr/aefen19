# infer.py
# 26.3.4
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from transformers import TextIteratorStreamer
from threading import Thread
from config import *

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

print("加载模型...")
kwargs = {"trust_remote_code": True}

if bnb_config:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=bnb_config,
        **kwargs
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        **kwargs
    )

if os.path.isdir(ADAPTER_DIR):
    model = PeftModel.from_pretrained(model, ADAPTER_DIR, device_map="auto")

model.eval()

def build_prompt(history):
    msgs = [SYSTEM_MSG] + history
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )

def generate_stream(user_text, history):
    history.append({"role": "user", "content": user_text})

    prompt = build_prompt(history)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
        use_cache=True
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    resp = ""
    for new_text in streamer:
        resp += new_text
        yield new_text

    history.append({"role": "assistant", "content": resp})
