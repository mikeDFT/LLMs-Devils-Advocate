import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from impl7.config import (
    BASE_MODEL, MAX_SEQ_LENGTH, MAX_PROMPT_LENGTH, LORA_RANK, LORA_ALPHA, LORA_TARGET_MODULES,
    DPO_BETA, DPO_LEARNING_RATE, DPO_BATCH_SIZE, DPO_GRADIENT_ACCUMULATION,
    DPO_MAX_STEPS, MIN_JUDGE_SCORE,
)
from impl7.dpo.dpo_settings import CURRENT_ITERATION, get_dpo_paths

dpo_paths = get_dpo_paths()
INPUT_MODEL_DIR = dpo_paths["input_model_dir"]
OUTPUT_MODEL_DIR = dpo_paths["output_model_dir"]
PREFERENCE_DATASET_PATH = dpo_paths["preference_dataset"]


def load_preference_data(path, min_score):
    """Load preference pairs, filtering out those where both responses are weak."""
    data = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            score_chosen = row.get("score_chosen")
            if score_chosen is not None and score_chosen < min_score:
                skipped += 1
                continue

            data.append({
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            })

    print(f"Loaded {len(data)} preference pairs (filtered {skipped} below score {min_score})")
    return data


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("DPO training requires a CUDA GPU")

    if not INPUT_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Input model not found at {INPUT_MODEL_DIR}. "
            "Please ensure the required SFT or older DPO adapter exists."
        )

    if not PREFERENCE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Preference dataset not found at {PREFERENCE_DATASET_PATH}. "
            "Run data_prep/generate_dpo_responses.py then data_prep/judge_dpo_responses.py first."
        )

    from unsloth import FastLanguageModel, PatchDPOTrainer
    from unsloth.chat_templates import get_chat_template
    from trl import DPOConfig, DPOTrainer
    
    PatchDPOTrainer()

    # Load the SFT or DPO-vN model
    print(f"Loading input model from {INPUT_MODEL_DIR}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(INPUT_MODEL_DIR),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
    )

    from datasets import Dataset
    raw_data = load_preference_data(PREFERENCE_DATASET_PATH, min_score=MIN_JUDGE_SCORE)
    ds = Dataset.from_list(raw_data)

    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_args = DPOConfig(
        beta=DPO_BETA,
        loss_type="sigmoid",
        per_device_train_batch_size=DPO_BATCH_SIZE,
        gradient_accumulation_steps=DPO_GRADIENT_ACCUMULATION,
        learning_rate=DPO_LEARNING_RATE,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        num_train_epochs=1,
        max_steps=DPO_MAX_STEPS,
        logging_steps=10,
        save_steps=80,
        save_total_limit=2,
        warmup_ratio=0.1,
        optim="adamw_8bit",
        output_dir=str(OUTPUT_MODEL_DIR),
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    model.save_pretrained(str(OUTPUT_MODEL_DIR))
    tokenizer.save_pretrained(str(OUTPUT_MODEL_DIR))
    print(f"DPO adapter saved to {OUTPUT_MODEL_DIR}")


if __name__ == "__main__":
    main()
