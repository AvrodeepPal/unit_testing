#unit_testing/llm_test/falcon.py

import genaius
from transformers import AutoTokenizer
import torch

model = genaius.model("Falcon-H1-34B-Base", "falcon::model_weights")

tokenizer = model.tokenizer

prompt = "The following is an educational article.Title: What is a Quantum Computer?Quantum computers are"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=350)
# hello
print(tokenizer.decode(output[0], skip_special_tokens=True))
