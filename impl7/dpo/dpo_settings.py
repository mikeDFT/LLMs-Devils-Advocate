from pathlib import Path

DPO_DIR = Path(__file__).parent
IMPLEMENTATION_ROOT = DPO_DIR.parent
DATA_DIR = IMPLEMENTATION_ROOT.parent / "data"

# Change this value to progress through iterations (1-indexed)
# Iteration 1 uses V0 (SFT) to generate responses and creates DPO_V1
# Iteration 2 uses V1 (DPO_V1) to generate responses and creates DPO_V2
CURRENT_ITERATION = 1

ITERATION_SETTINGS = [
    { # Iteration 1 -> output DPO V1
        "DPO_PROMPT_COUNT": 50,
        "TEMPERATURE": 0.8,
    },
    { # Iteration 2 -> output DPO V2
        "DPO_PROMPT_COUNT": 300,
        "TEMPERATURE": 0.8,
    },
    { # Iteration 3 -> output DPO V3
        "DPO_PROMPT_COUNT": 500,
        "TEMPERATURE": 0.4,
    },
]

def get_dpo_paths(iteration=CURRENT_ITERATION):
    """Returns dynamic paths for a given DPO iteration."""
    dpo_data_dir = DATA_DIR / "dpo_iterations" / f"v{iteration}"
    
    input_model_dir = (
        IMPLEMENTATION_ROOT / "sft_adapter" if iteration == 1 
        else IMPLEMENTATION_ROOT / f"dpo_adapter_v{iteration-1}"
    )
    output_model_dir = IMPLEMENTATION_ROOT / f"dpo_adapter_v{iteration}"
    
    # Ensure dataset directory exists
    dpo_data_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "raw_responses": dpo_data_dir / "dpo_raw_responses.json",
        "judge_progress": dpo_data_dir / "dpo_judge_progress.json",
        "preference_dataset": dpo_data_dir / "preference_dataset.jsonl",
        "input_model_dir": input_model_dir,
        "output_model_dir": output_model_dir,
        "settings": ITERATION_SETTINGS[iteration - 1]
    }

def get_dataset_offset(iteration=CURRENT_ITERATION):
    """Computes how many unseen prompts to skip based on sums of older iterations."""
    offset = 0
    for i in range(1, iteration):
        offset += ITERATION_SETTINGS[i - 1]["DPO_PROMPT_COUNT"]
    return offset
