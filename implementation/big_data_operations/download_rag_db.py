import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

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
    
    print(f"Downloading RAG database from {repo_id}")
    print(f"Target directory: {local_dir}")
    
    RAG_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    # pulls entire repo's files
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type="dataset",
        token=hf_token
    )
    
    print("Downloaded RAG database successfully.")

if __name__ == "__main__":
    main()
