#unit_testing/llm_test/deepseek.py

import genaius
import torch
import transformers

model = genaius.model("deepseek-coder-33b-instruct", "deepseek::model_weights")
tokenizer = model.tokenizer

prompt = "### Instruction:\ngenerate a python code snippet for 2 sum problem\n### Response:\n"

inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)

model.eval()
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

input_len = inputs['input_ids'].shape[1]
generated = output[0][input_len:]

# Decode and clean up the BPE artifacts
text = tokenizer.decode(generated, skip_special_tokens=True)
text = text.replace('Ġ', ' ').replace('Ċ', '\n')

print(text)