import json
from pathlib import Path

# Paths to your current dataset and the new, fixed dataset
INPUT_PATH = Path("data/synthetic_dpo_dataset.jsonl")
OUTPUT_PATH = Path("data/synthetic_dpo_dataset_out.jsonl")

def fix_no_fallacy_rows():
    fixed_count = 0
    total_count = 0
    
    # Open the current dataset for reading, and a new one for writing
    with open(INPUT_PATH, "r", encoding="utf-8") as infile, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            total_count += 1
            data = json.loads(line)
            
            # Check if the 'fallacy_analysis' key is missing
            if "fallacy_analysis" not in data:
                # Inject the missing key exactly as you requested
                data["fallacy_analysis"] = "Fallacy Detected: no_fallacy."
                fixed_count += 1
                
            # Write the row (whether it was modified or not) to the new file
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    print("=" * 40)
    print("Dataset Patch Complete")
    print("=" * 40)
    print(f"Total Rows Processed: {total_count}")
    print(f"Rows Patched with 'no_fallacy': {fixed_count}")
    print(f"New File Saved To: {OUTPUT_PATH}")

if __name__ == "__main__":
    fix_no_fallacy_rows()
