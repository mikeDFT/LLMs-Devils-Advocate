import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GGUF_OUTPUT_DIR

def main():
    # load HF_ACCESS_KEY from the .env file
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    
    hf_token = os.environ.get("HF_ACCESS_KEY")
    if not hf_token:
        raise ValueError("HF_ACCESS_KEY not found in env vars. Add to .env file.")
        
    repo_id = "andrada28/devils-advocate-dpo-v1-gguf"
    local_dir = str(GGUF_OUTPUT_DIR)
    
    print(f"Downloading GGUF model from {repo_id}")
    print(f"Target directory: {local_dir}")
    
    GGUF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # pulls entire repo's files
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=hf_token
    )
    
    print(f"Downloaded. The GGUF file is available in {local_dir}")

if __name__ == "__main__":
    main()
