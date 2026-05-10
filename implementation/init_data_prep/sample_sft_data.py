import json
import random
from pathlib import Path

INPUT_PATH = Path("data/synthetic_dpo_dataset.jsonl")

def sample_sft_data(n: int = 15):
    """
    Reads the dataset at INPUT_PATH and prints n randomly selected rows.
    """
    if not INPUT_PATH.exists():
        print(f"Error: Dataset file not found at {INPUT_PATH}")
        return

    print(f"Reading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_rows = len(lines)
    if total_rows == 0:
        print("The dataset is empty.")
        return

    # Adjust n if it's larger than the dataset
    num_to_sample = min(n, total_rows)
    
    print(f"Sampling {num_to_sample} random rows from a total of {total_rows}...")
    sampled_lines = random.sample(lines, num_to_sample)

    for i, line in enumerate(sampled_lines, 1):
        try:
            data = json.loads(line)
            print(f"\n[SAMPLE {i}]")
            keys_to_print = ["user_premise", "fallacy_analysis", "chosen_response", "rag_context"]
            subset = {k: data.get(k, "N/A") for k in keys_to_print}
            
            print(json.dumps(subset, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(f"\n[SAMPLE {i}] Error: Could not decode JSON line.")
            print(f"Raw line: {line.strip()[:200]}...")

if __name__ == "__main__":
    sample_sft_data(5)
