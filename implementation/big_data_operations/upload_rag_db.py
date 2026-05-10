import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAG_DB_DIR

def main():
    # load HF_ACCESS_KEY from the .env file
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    
    hf_token = os.environ.get("HF_ACCESS_KEY")
    if not hf_token:
        raise ValueError("HF_ACCESS_KEY not found in env vars. Add to .env file.")
        
    repo_id = "MikeDFT/devils_advocate_rag_lancedb"
    local_dir = str(RAG_DB_DIR)
    
    print(f"Uploading RAG database from {local_dir}")
    print(f"Target repository: {repo_id}")
    
    if not RAG_DB_DIR.exists():
        raise FileNotFoundError(f"Directory {local_dir} does not exist. Nothing to upload.")
    
    api = HfApi()
    
    # Create the private dataset repo if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, token=hf_token, exist_ok=True)
    
    # Upload the folder
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token
    )
    
    print("Upload complete.")

if __name__ == "__main__":
    main()
