#unit_testing/llm_test/phi.py

import genaius
from transformers import AutoTokenizer
import torch

model = genaius.model("Phi-4-mini-reasoning", "phi::model_weights")
tokenizer = model.tokenizer

# Fix tokenizer padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "explain in brief about blackhole"
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)

model.eval()
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=1000,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))