#unit_testing/llm_test/mistral.py

import genaius
from transformers import AutoTokenizer
import torch

model = genaius.model("Mistral-7B-Instruct-v0.3", "mistral::model_weights")

tokenizer = model.tokenizer

prompt = "explain in brief about blackhole"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=350)
# hello
print(tokenizer.decode(output[0], skip_special_tokens=True))
