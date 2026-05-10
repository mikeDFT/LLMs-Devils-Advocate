"""
Pushes the generated synthetic DPO dataset to Hugging Face Hub.
"""

import os
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

# Add implementation to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

def main():
    load_dotenv()
    
    hf_key = os.getenv("HF_ACCESS_KEY")
    if not hf_key:
        raise ValueError("HF_ACCESS_KEY not found in .env file.")
        
    dataset_path = DATA_DIR / "sft_dataset_split.jsonl"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    print(f"Loading {dataset_path.name}...")
    dataset = load_dataset('json', data_files=str(dataset_path), split="train")
    
    print(f"Loaded {len(dataset)} rows.")
    print("Pushing...")
    
    dataset.push_to_hub(
        "MikeDFT/devils-advocate-sft", 
        token=hf_key,
        private=False # Set to True if you want the dataset to be private
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
