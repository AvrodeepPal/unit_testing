#unit_testing/llm_test/llama.py

import genaius
from transformers import AutoTokenizer
import torch
import safetensors
import bitsandbytes
import accelerate

model = genaius.model("Llama-3.1-70B-Instruct", "llama::model_weights")

tokenizer = model.tokenizer

prompt = "explain in brief about blackhole"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=350)
# hello
print(tokenizer.decode(output[0], skip_special_tokens=True))
