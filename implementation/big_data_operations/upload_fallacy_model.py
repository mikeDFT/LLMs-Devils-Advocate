import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FALLACY_MODEL_DIR

def main():
    # load HF_ACCESS_KEY from the .env file
    # We find .env in the project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    
    hf_token = os.environ.get("HF_ACCESS_KEY")
    if not hf_token:
        raise ValueError("HF_ACCESS_KEY not found in env vars. Add to .env file.")
        
    repo_id = "MikeDFT/devils-advocate-fallacy-model"
    local_dir = str(FALLACY_MODEL_DIR)
    
    print(f"Uploading fallacy model from {local_dir}")
    print(f"Target repository: {repo_id}")
    
    if not FALLACY_MODEL_DIR.exists():
        raise FileNotFoundError(f"Directory {local_dir} does not exist. Nothing to upload.")
    
    api = HfApi()
    
    # Create the private model repo if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, token=hf_token, exist_ok=True)
    
    # Upload only the specific files required by detector.py
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
        allow_patterns=[
            "config.json", 
            "model.safetensors", 
            "special_tokens_map.json", 
            "tokenizer.json", 
            "tokenizer_config.json"
        ]
    )
    
    print("Upload complete.")

if __name__ == "__main__":
    main()
