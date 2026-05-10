import json
from pathlib import Path
import sys

# Add project root to sys.path to allow importing from implementation.config
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from implementation.config import FALLACY_LABELS

# Paths to your current dataset and the new dataset
INPUT_PATH = Path("data/synthetic_dpo_dataset.jsonl")
OUTPUT_PATH = Path("data/synthetic_dpo_dataset_out.jsonl")

def replace_fallacy_labels():
    processed_count = 0
    modified_count = 0
    
    # Pre-compute the replacements from FALLACY_LABELS
    # e.g., 'appeal_to_emotion' -> 'appeal to emotion'
    replacements = {label: label.replace('_', ' ') for label in FALLACY_LABELS if '_' in label}
    replacements["user_premise"] = "user premise"
    
    with open(INPUT_PATH, "r", encoding="utf-8") as infile, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            processed_count += 1
            data = json.loads(line)
            
            was_modified = False
            
            # Check and replace in chosen_response
            if "chosen_response" in data:
                text = data["chosen_response"]
                original_text = text
                
                # Replace each fallacy label with its normal words counterpart
                for label, normal_words in replacements.items():
                    if label in text:
                        print(f"{text}\n{label}\n")
                        text = text.replace(label, normal_words)
                
                # If changes were made, update the dictionary
                if text != original_text:
                    data["chosen_response"] = text
                    was_modified = True
                    
            if was_modified:
                modified_count += 1
                
            # Write the row (whether it was modified or not) to the new file
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    print("=" * 40)
    print("Fallacy Labels Replacement Complete")
    print("=" * 40)
    print(f"Total Rows Processed: {processed_count}")
    print(f"Rows Modified: {modified_count}")
    print(f"New File Saved To: {OUTPUT_PATH}")


if __name__ == "__main__":
    replace_fallacy_labels()
