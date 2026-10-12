#unit_testing/llm_test/gemma.py

import genaius
import torch
import transformers
import torchvision
import PIL
# 1. Load the model wrapper
model_wrapper = genaius.model("gemma-4-31B-it", "gemma::model_weights")

# 2. Extract the underlying model and processor
actual_model = model_wrapper.model
processor = model_wrapper.processor

# 3. Set up the conversation (Gemma uses a chat template)
system_prompt = "You are a helpful assistant."
user_prompt = "explain in brief about machine learning"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": user_prompt},
]

# 4. Apply the chat template (no "thinking" or extra flags)
formatted_text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 5. Tokenize and move to the model's device
inputs = processor(text=formatted_text, return_tensors="pt").to(actual_model.device)

# 6. Generate with sensible parameters
actual_model.eval()
with torch.no_grad():
    output = actual_model.generate(
        **inputs,
        max_new_tokens=350,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

# 7. Decode only the newly generated tokens (skip the prompt)
input_len = inputs['input_ids'].shape[1]
generated_tokens = output[0][input_len:]
text = processor.decode(generated_tokens, skip_special_tokens=True)

# 8. Clean up any BPE artifacts (optional)
text = text.replace('Ġ', ' ').replace('Ċ', '\n')

print(text)