"""
Export the DPO-aligned model to GGUF for LMStudio inference.

Merges LoRA adapters into the base model, then quantizes to Q4_K_M GGUF.

Output: implementation/gguf_output/
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DPO_ADAPTER_DIR, SFT_ADAPTER_DIR, GGUF_OUTPUT_DIR, MAX_SEQ_LENGTH


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GGUF export requires a CUDA GPU")

    # Prefer DPO adapter if it exists, otherwise fall back to SFT
    adapter_dir = DPO_ADAPTER_DIR if DPO_ADAPTER_DIR.exists() else SFT_ADAPTER_DIR
    if not adapter_dir.exists():
        raise FileNotFoundError(
            "No trained adapter found. Run sft_train.py and/or dpo_train.py first."
        )

    from unsloth import FastLanguageModel

    print(f"Loading adapter from {adapter_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    GGUF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print("Exporting to Q4_K_M GGUF... (this may take a while)")
        model.save_pretrained_gguf(
            str(GGUF_OUTPUT_DIR),
            tokenizer,
            quantization_method="q4_k_m"
        )
        print(f"GGUF model saved to {GGUF_OUTPUT_DIR}")

    except Exception as e:
        print(f"GGUF export failed: {e}")

if __name__ == "__main__":
    main()
