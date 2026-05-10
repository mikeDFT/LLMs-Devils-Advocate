import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SFT_ADAPTER_DIR

def main():
    # load HF_ACCESS_KEY from the .env file
    load_dotenv()
    
    hf_token = os.environ.get("HF_ACCESS_KEY")
    if not hf_token:
        raise ValueError("HF_ACCESS_KEY not found in env vars. Add to .env file.")
        
    repo_id = "MikeDFT/devils-advocate-adapter-gen2"
    local_dir = str(SFT_ADAPTER_DIR)
    
    print(f"Downloading adapter from {repo_id}")
    print(f"Target directory: {local_dir}")
    
    SFT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    
    # pulls entire repo's files
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=hf_token
    )
    
    print("Downloaded. Run: py implementation/big_data_operations/export_gguf.py")

if __name__ == "__main__":
    main()
