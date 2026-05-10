"""
Filters the generated synthetic DPO dataset to remove rows where the
chosen_response lacks sufficient evidence (quantitative or qualitative).

CURRENTLY NOT USED BECAUSE TOO AGGRESSIVE.
"""

import json
import re
import sys
from pathlib import Path

# Add implementation to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

def has_evidence(response):
    # Named author + year citation pattern e.g. "Smith (2010)" or "Smith et al."
    has_author_year = bool(re.search(r'[A-Z][a-z]+.*?\b(19|20)\d{2}\b', response))
    
    # Named institution or publication
    has_institution = bool(re.search(
        r'\b(University|Institute|Journal|Amnesty|Council|Center|Centre|IPCC|NSF|EPA|WHO)\b',
        response, re.IGNORECASE
    ))
    
    # Quantitative evidence
    has_numbers = bool(re.search(r'\b\d+\s*(%|percent|billion|million|thousand)\b', response, re.IGNORECASE))
    has_specific_stat = bool(re.search(r'\b\d{1,3}[,.]?\d+\b', response))
    
    # Direct quote markers
    has_quote = bool(re.search(r'["\u201c\u201d].*?["\u201c\u201d]', response))
    
    quantitative = has_numbers or has_specific_stat
    qualitative = has_author_year or has_institution or has_quote
    
    # Pass if quantitative OR qualitative — not requiring both
    return quantitative or qualitative


def main():
    dataset_path = DATA_DIR / "synthetic_dpo_dataset.jsonl"
    rejected_path = DATA_DIR / "synthetic_dpo_dataset_rejected.jsonl"
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    print(f"Loading dataset from {dataset_path}...")
    
    kept_rows = []
    rejected_rows = []
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            response = data.get("chosen_response", "")
            
            if has_evidence(response):
                kept_rows.append(data)
            else:
                rejected_rows.append(data)
                
    print(f"Filtering complete.")
    print(f"Kept rows: {len(kept_rows)}")
    print(f"Rejected rows: {len(rejected_rows)}")
    
    # Safely overwrite the original file with the kept rows
    with open(dataset_path, "w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    # Save the rejected rows to a separate file for inspection
    if rejected_rows:
        with open(rejected_path, "w", encoding="utf-8") as f:
            for row in rejected_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Rejected rows saved for inspection to: {rejected_path.name}")
        
    print(f"Original dataset successfully overwritten with only high-quality rows.")


if __name__ == "__main__":
    main()
