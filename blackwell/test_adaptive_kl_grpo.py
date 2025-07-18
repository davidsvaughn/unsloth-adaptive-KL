"""
Example script demonstrating the new adaptive KL functionality for GRPO training.
This script shows how to use the different KL adaptation methods to prevent
pathological length exploitation in GRPO training.
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
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

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

# Demonstrate different adaptive KL methods
from trl import GRPOConfig, GRPOTrainer

print("\n" + "="*50)
print("DEMO: No adaptive KL (baseline)")
print("="*50)

training_args_baseline = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=4,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    max_steps=3,
    save_steps=100,
    report_to="none",
    output_dir="outputs_baseline",
    # No adaptive KL - using default 'none'
)

trainer_baseline = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[match_format_exactly, check_answer],
    args=training_args_baseline,
    train_dataset=dataset,
)

print("Training with baseline (no adaptive KL)...")
trainer_baseline.train()

print("\n" + "="*50)
print("DEMO: KL Clipping Method")
print("="*50)

training_args_clip = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=4,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    max_steps=3,
    save_steps=100,
    report_to="none",
    output_dir="outputs_clip",
    # Adaptive KL: Clipping method
    kl_adaptation_method="clip",
    kl_clip_threshold=0.5,  # Clip KL at 0.5
)

trainer_clip = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[match_format_exactly, check_answer],
    args=training_args_clip,
    train_dataset=dataset,
)

print("Training with KL clipping (threshold=0.5)...")
trainer_clip.train()

print("\n" + "="*50)
print("DEMO: Length-Normalized KL")
print("="*50)

training_args_length = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=4,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    max_steps=3,
    save_steps=100,
    report_to="none",
    output_dir="outputs_length_norm",
    # Adaptive KL: Length normalization
    kl_adaptation_method="length_normalized",
)

trainer_length = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[match_format_exactly, check_answer],
    args=training_args_length,
    train_dataset=dataset,
)

print("Training with length-normalized KL...")
trainer_length.train()

print("\n" + "="*50)
print("DEMO: Dynamic Beta Scheduling")
print("="*50)

training_args_dynamic = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=4,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    max_steps=5,  # More steps to see dynamic behavior
    save_steps=100,
    report_to="none",
    output_dir="outputs_dynamic",
    # Adaptive KL: Dynamic beta scheduling
    kl_adaptation_method="dynamic",
    dynamic_beta_decay=0.8,  # Decay beta by 20% each time
    reward_threshold=1.0,    # When average reward > 1.0
    beta_min=0.001,         # Minimum beta value
    beta_schedule_steps=3,  # After 3 steps, set beta to minimum
)

trainer_dynamic = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[match_format_exactly, check_answer],
    args=training_args_dynamic,
    train_dataset=dataset,
)

print("Training with dynamic beta scheduling...")
trainer_dynamic.train()

print("\n" + "="*50)
print("DEMO: Sigmoid KL Clipping")
print("="*50)

training_args_sigmoid = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=4,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    max_steps=3,
    save_steps=100,
    report_to="none",
    output_dir="outputs_sigmoid",
    # Adaptive KL: Sigmoid clipping
    kl_adaptation_method="sigmoid",
    kl_target=0.2,  # Target KL divergence
)

trainer_sigmoid = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[match_format_exactly, check_answer],
    args=training_args_sigmoid,
    train_dataset=dataset,
)

print("Training with sigmoid KL clipping (target=0.2)...")
trainer_sigmoid.train()

print("\n" + "="*50)
print("ADAPTIVE KL DEMO COMPLETE!")
print("="*50)
print("Methods demonstrated:")
print("1. Baseline (no adaptation)")
print("2. KL Clipping - clips KL values at threshold")
print("3. Length Normalization - normalizes KL by sequence length")
print("4. Dynamic Beta - adjusts beta based on reward convergence")
print("5. Sigmoid Clipping - soft clipping with sigmoid function")
print()
print("Each method helps prevent pathological length exploitation")
print("where models generate longer sequences to reduce KL divergence.")
print("="*50)
