#unit_testing/finetune_test/test.py

import torch
import os
import tempfile

_cache = os.path.join(tempfile.gettempdir(), "torch_cache")
os.makedirs(_cache, exist_ok=True)

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _cache)
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cache, "triton"))
os.environ.setdefault("HF_HOME", os.path.join(_cache, "hf"))
os.environ.setdefault("XDG_CACHE_HOME", _cache)

# getpass.getuser() fallback for anonymous UIDs
os.environ.setdefault("USER", "sandbox")
os.environ.setdefault("LOGNAME", "sandbox")


import json
import inspect

import accelerate

from pathlib import Path
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

DATA_PATH = Path(__file__).parent / "indicqa.bn.json"

OUTPUT_DIR=Path(__file__).parent /"adapter"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))

GRADIENT_ACCUMULATION = int(
    os.getenv("GRADIENT_ACCUMULATION", "4")
)

LR = float(os.getenv("LR", "5e-5"))

EPOCHS = float(os.getenv("EPOCHS", "3"))

MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "2000"))

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))

MAX_STEPS = int(os.getenv("MAX_STEPS", "-1"))

VAL_SPLIT = float(os.getenv("VAL_SPLIT", "0.05"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================================================
# GPU CHECK
# =========================================================

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: No GPU detected")

# =========================================================
# LOAD TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

# =========================================================
# LOAD MODEL
# =========================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
    if torch.cuda.is_available()
    else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

# =========================================================
# MEMORY OPTIMIZATION
# =========================================================

model.gradient_checkpointing_enable()

model.config.use_cache = False

# =========================================================
# TOKENIZER FIXES
# =========================================================

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# =========================================================
# LOAD DATASET
# =========================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}"
    )

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_instruction_output_records(data):

    # =====================================================
    # FORMAT A
    # =====================================================

    if isinstance(data, list):

        if (
            data
            and isinstance(data[0], dict)
            and "instruction" in data[0]
            and "output" in data[0]
        ):
            return data

        return []

    # =====================================================
    # FORMAT B (SQuAD STYLE)
    # =====================================================

    if isinstance(data, dict) and "data" in data:

        records = []

        for article in data.get("data", []):

            for paragraph in article.get(
                "paragraphs",
                []
            ):

                context = (
                    paragraph.get("context") or ""
                ).strip()

                for qa in paragraph.get("qas", []):

                    question = (
                        qa.get("question") or ""
                    ).strip()

                    answers = qa.get("answers") or []

                    answer_text = ""

                    for ans in answers:

                        text = (
                            ans.get("text") or ""
                        ).strip()

                        if text:
                            answer_text = text
                            break

                    if not question or not answer_text:
                        continue

                    if context:

                        instruction = (
                            f"Context: {context}\n"
                            f"Question: {question}"
                        )

                    else:
                        instruction = question

                    records.append({
                        "instruction": instruction,
                        "output": answer_text,
                    })

        return records

    return []

raw_payload = load_json(DATA_PATH)

raw_data = to_instruction_output_records(
    raw_payload
)

if not raw_data:
    raise ValueError(
        "No trainable samples found"
    )

if MAX_SAMPLES > 0:
    raw_data = raw_data[:MAX_SAMPLES]

print(f"📦 Loaded {len(raw_data)} samples")

# =========================================================
# BUILD CHAT DATASET
# =========================================================

dataset = []

for d in raw_data:

    messages = [
        {
            "role": "user",
            "content": d["instruction"],
        },
        {
            "role": "assistant",
            "content": d["output"],
        }
    ]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    dataset.append({
        "text": full_text,
        "answer": d["output"],
    })

hf_dataset = Dataset.from_list(dataset)

# =========================================================
# TOKENIZATION WITH LABEL MASKING
# =========================================================

def tokenize(example):

    messages = [
        {
            "role": "user",
            "content": example["text"],
        }
    ]

    # FULL CONVERSATION
    full_tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = full_tokens["input_ids"]

    attention_mask = full_tokens["attention_mask"]

    labels = input_ids.copy()

    # =====================================================
    # MASK EVERYTHING BEFORE ASSISTANT RESPONSE
    # =====================================================

    assistant_text = example["answer"]

    assistant_tokens = tokenizer(
        assistant_text,
        add_special_tokens=False,
    )["input_ids"]

    assistant_len = len(assistant_tokens)

    mask_until = len(input_ids) - assistant_len

    labels[:mask_until] = [-100] * mask_until

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

tokenized = hf_dataset.map(
    tokenize,
    batched=False,
    remove_columns=["text", "answer"],
)

# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

split_dataset = tokenized.train_test_split(
    test_size=VAL_SPLIT,
    seed=42,
)

train_dataset = split_dataset["train"]

eval_dataset = split_dataset["test"]

print("Train samples:", len(train_dataset))

print("Validation samples:", len(eval_dataset))

# =========================================================
# LORA CONFIG
# =========================================================

peft_config = LoraConfig(

    task_type=TaskType.CAUSAL_LM,

    r=32,

    lora_alpha=64,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    lora_dropout=0.05,

    bias="none",
)

# =========================================================
# ATTACH LORA
# =========================================================

lora_model = get_peft_model(
    model,
    peft_config
)

lora_model.print_trainable_parameters()

# =========================================================
# TRAINING ARGS
# =========================================================

training_args_kwargs = {

    "output_dir": OUTPUT_DIR,

    "num_train_epochs": EPOCHS,

    "max_steps": MAX_STEPS,

    "per_device_train_batch_size":
        BATCH_SIZE,

    "per_device_eval_batch_size":
        BATCH_SIZE,

    "gradient_accumulation_steps":
        GRADIENT_ACCUMULATION,

    "learning_rate": LR,

    "fp16": torch.cuda.is_available(),

    "logging_steps": 10,

    "eval_strategy": "steps",

    "eval_steps": 50,

    "save_strategy": "steps",

    "save_steps": 50,

    "save_total_limit": 2,

    "load_best_model_at_end": True,

    "report_to": "none",

    "disable_tqdm": False,
}

# =========================================================
# VERSION COMPATIBILITY
# =========================================================

supported_args = set(
    inspect.signature(
        TrainingArguments.__init__
    ).parameters.keys()
)

filtered_training_args_kwargs = {

    key: value

    for key, value
    in training_args_kwargs.items()

    if key in supported_args
}

training_args = TrainingArguments(
    **filtered_training_args_kwargs
)

# =========================================================
# TRAINER
# =========================================================

trainer = Trainer(

    model=lora_model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,
)

# =========================================================
# TRAIN
# =========================================================

print("\n🚀 Starting training...\n")

try:

    trainer.train()

except KeyboardInterrupt:

    print(
        "\n🛑 Interrupted. Saving adapter...\n"
    )

    lora_model.save_pretrained(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

# =========================================================
# SAVE
# =========================================================

print("\n💾 Saving final adapter...\n")

lora_model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("\n✅ Training complete!\n")