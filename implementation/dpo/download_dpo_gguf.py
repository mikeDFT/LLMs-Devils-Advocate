import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).parent.parent))

from impl7.config import GGUF_OUTPUT_DIR


def main():
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    hf_token = os.environ.get("HF_ACCESS_KEY")
    if not hf_token:
        raise ValueError("HF_ACCESS_KEY not found in .env")

    repo_id = "andrada28/devils-advocate-dpo-v1-gguf"

    local_dir = GGUF_OUTPUT_DIR / "dpo_v1"

    print(f"Downloading DPO GGUF from {repo_id}")
    print(f"Target directory: {local_dir}")

    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=hf_token,
    )

    print(f"Downloaded. GGUF is available in {local_dir}")


if __name__ == "__main__":
    main()