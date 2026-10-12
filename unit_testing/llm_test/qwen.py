#unit_testing/llm_test/qwen.py

import genaius
from transformers import AutoTokenizer
import torch

model = genaius.model("Qwen2.5-14B-Instruct", "qwen25::model_weights")

tokenizer = model.tokenizer

prompt = "explain in brief about blackhole"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=350)
# hello
print(tokenizer.decode(output[0], skip_special_tokens=True))
