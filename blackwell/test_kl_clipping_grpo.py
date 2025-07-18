"""
Simple KL clipping demonstration for GRPO training.
Shows baseline (no clipping), hard clipping, and soft clipping methods.
"""

from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-0.6B-Base",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.7,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Setup chat template and dataset
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

chat_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + eos_token }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{{ '{system_prompt}' + eos_token }}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"
    "{% endif %}"
)

chat_template = chat_template.replace(
    "'{system_prompt}'", f"'{system_prompt}'"
).replace("'{reasoning_start}'", f"'{reasoning_start}'")
tokenizer.chat_template = chat_template

# Load and prepare dataset
from datasets import load_dataset
import pandas as pd
import numpy as np

dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
dataset = dataset.to_pandas()[["expected_answer", "problem", "generated_solution"]]

is_number = pd.to_numeric(
    pd.Series(dataset["expected_answer"]), errors="coerce"
).notnull()
dataset = dataset.iloc[np.where(is_number)[0]]

def format_dataset(x):
    expected_answer = x["expected_answer"]
    problem = x["problem"]
    thoughts = x["generated_solution"]
    thoughts = thoughts.replace("<think>", "").replace("</think>", "")
    thoughts = thoughts.strip()
    final_prompt = (
        reasoning_start
        + thoughts
        + reasoning_end
        + solution_start
        + expected_answer
        + solution_end
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": final_prompt},
    ]

dataset["Messages"] = dataset.apply(format_dataset, axis=1)
dataset["N"] = dataset["Messages"].apply(
    lambda x: len(tokenizer.apply_chat_template(x))
)
dataset = dataset.loc[dataset["N"] <= max_seq_length / 2].copy()

from datasets import Dataset
dataset["text"] = tokenizer.apply_chat_template(
    dataset["Messages"].values.tolist(), tokenize=False
)
dataset = Dataset.from_pandas(dataset)

# Initial SFT training
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    ),
)

trainer.train()

# Prepare GRPO dataset
del dataset
torch.cuda.empty_cache()
import gc
gc.collect()

from datasets import load_dataset
dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")

def extract_hash_answer(text):
    return text

dataset = dataset.map(
    lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["prompt"]},
        ],
        "answer": extract_hash_answer(x["solution"]),
    }
)

# Reward functions
import re

solution_end_regex = (
    r"</SOLUTION>[\s]{0,}" + "(?:" + re.escape(tokenizer.eos_token) + ")?"
)

match_format = re.compile(
    rf"{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end_regex}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1) if (guess := match_format.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(-2.0)
            continue
        if guess == true_answer:
            score += 5.0
        elif guess.strip() == true_answer.strip():
            score += 3.5
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1:
                    score += 2.0
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 1.5
                else:
                    score -= 2.5
            except:
                score -= 4.5
        scores.append(score)
    return scores

# Prepare tokenized dataset
tokenized = dataset.map(
    lambda x: {
        "tokens": tokenizer.apply_chat_template(
            x["prompt"], add_generation_prompt=True, tokenize=True
        )
    },
    batched=True,
)

tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

import numpy as np
maximum_length = int(np.quantile(tokenized["L"], 0.9))
print("Max Length = ", maximum_length)

dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
del tokenized

max_prompt_length = maximum_length + 1
max_completion_length = max_seq_length - max_prompt_length

from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

# Common training arguments
from trl import GRPOConfig, GRPOTrainer

common_args = {
    "vllm_sampling_params": vllm_sampling_params,
    "temperature": 1.0,
    "learning_rate": 5e-6,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "linear",
    "optim": "adamw_8bit",
    "logging_steps": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "num_generations": 4,
    "max_prompt_length": max_prompt_length,
    "max_completion_length": max_completion_length,
    "max_steps": 3,
    "save_steps": 100,
    "report_to": "none",
}

print("\n" + "="*60)
print("KL CLIPPING DEMONSTRATION WITH PER-TOKEN THRESHOLDS")
print("="*60)

# 1. Baseline (no clipping)
print("\n1. BASELINE - No KL clipping")
print("-" * 30)

training_args_baseline = GRPOConfig(
    output_dir="outputs_baseline",
    kl_clip_method="none",  # No clipping
    **common_args
)

trainer_baseline = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[match_format_exactly, check_answer],
    args=training_args_baseline,
    train_dataset=dataset,
)

print("Training with baseline (no KL clipping)...")
trainer_baseline.train()

# 2. Per-token soft clipping (recommended)
print("\n2. PER-TOKEN SOFT CLIPPING - Scaled by reference length")
print("-" * 30)

training_args_per_token = GRPOConfig(
    output_dir="outputs_per_token",
    kl_clip_method="soft",           # Soft clipping
    kl_clip_threshold=0.1,                # 0.1 nats per token
    use_per_token_kl_threshold=True,      # Scale by reference length
    **common_args
)

trainer_per_token = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[match_format_exactly, check_answer],
    args=training_args_per_token,
    train_dataset=dataset,
)

print("Training with per-token KL clipping (0.1 nats per token)...")
trainer_per_token.train()

# 3. Total KL threshold (legacy behavior)
print("\n3. TOTAL KL THRESHOLD - Direct threshold application")
print("-" * 30)

training_args_total = GRPOConfig(
    output_dir="outputs_total",
    kl_clip_method="soft",           # Soft clipping
    kl_clip_threshold=5.0,                # 5.0 total nats
    use_per_token_kl_threshold=False,     # Use threshold directly
    **common_args
)

trainer_total = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[match_format_exactly, check_answer],
    args=training_args_total,
    train_dataset=dataset,
)

print("Training with total KL clipping (5.0 total nats)...")
trainer_total.train()

# 4. Hard clipping with per-token threshold
print("\n4. PER-TOKEN HARD CLIPPING - Sharp cutoff scaled by length")
print("-" * 30)

training_args_hard = GRPOConfig(
    output_dir="outputs_hard_clip",
    kl_clip_method="hard",          # Hard clipping
    kl_clip_threshold=0.15,               # 0.15 nats per token
    use_per_token_kl_threshold=True,      # Scale by reference length
    **common_args
)

trainer_hard = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[match_format_exactly, check_answer],
    args=training_args_hard,
    train_dataset=dataset,
)

print("Training with hard per-token KL clipping (0.15 nats per token)...")
trainer_hard.train()

print("\n" + "="*60)
print("KL CLIPPING DEMONSTRATION COMPLETE!")
print("="*60)
print("Methods demonstrated:")
print("1. Baseline - No clipping (standard GRPO)")
print("2. Per-token soft clipping - F.softplus(KL - per_token_threshold * length)")
print("3. Total KL threshold - F.softplus(KL - total_threshold)")
print("4. Per-token hard clipping - max(0, KL - per_token_threshold * length)")
print("\nBenefits of per-token thresholds:")
print("- Intuitive threshold setting (nats per token)")
print("- Length-agnostic behavior")
print("- Transferable across different tasks")
print("- Prevents gaming by generating longer sequences")
print("="*60)
