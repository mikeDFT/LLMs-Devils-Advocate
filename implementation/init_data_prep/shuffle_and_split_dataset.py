import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RANDOM_SEED, DATA_DIR

MASTER_FILE = DATA_DIR / "synthetic_dpo_dataset.jsonl"
SFT_OUTPUT = DATA_DIR / "sft_dataset_split.jsonl"
DPO_OUTPUT = DATA_DIR / "dpo_dataset_split.jsonl"
DPO_RESERVE_SIZE = 500

def shuffle_and_split():
    with open(MASTER_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        
    print(f"Loaded {len(data)} total rows.")

    random.seed(RANDOM_SEED)
    random.shuffle(data)

    dpo_data = data[:DPO_RESERVE_SIZE]
    sft_data = data[DPO_RESERVE_SIZE:]

    with open(SFT_OUTPUT, "w", encoding="utf-8") as f:
        for row in sft_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    with open(DPO_OUTPUT, "w", encoding="utf-8") as f:
        for row in dpo_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 40)
    print("Dataset Split Complete!")
    print(f"SFT Data (Ready to train): {len(sft_data)} rows -> {SFT_OUTPUT}")
    print(f"DPO Base (Needs rejected responses): {len(dpo_data)} rows -> {DPO_OUTPUT}")

if __name__ == "__main__":
    shuffle_and_split()