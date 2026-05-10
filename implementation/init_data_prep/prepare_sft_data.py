"""
Prepare SFT training data from DebateSum CSV.

Loads the DebateSum dataset, filters for quality rows, and formats them as Qwen3 chat-template conversations for supervised fine-tuning.

Output: data/sft_dataset.jsonl
"""

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEBATESUM_CSV_PATH, SFT_DATASET_PATH, SYSTEM_PROMPT, DATA_DIR, RANDOM_SEED

TARGET_SAMPLES = 6000
MIN_EXTRACT_CHARS = 100
MAX_EXTRACT_CHARS = 3000
MIN_DOCUMENT_CHARS = 200
MAX_DOCUMENT_CHARS = 5000


def load_and_filter(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=["Extract", "Full-Document"])

    # Length filters ensure we get substantive but not overwhelming debate examples
    extract_len = df["Extract"].str.len()
    doc_len = df["Full-Document"].str.len()
    mask = (
        (extract_len >= MIN_EXTRACT_CHARS) & (extract_len <= MAX_EXTRACT_CHARS)
        & (doc_len >= MIN_DOCUMENT_CHARS) & (doc_len <= MAX_DOCUMENT_CHARS)
    )
    df = df[mask].reset_index(drop=True)
    print(f"After filtering: {len(df)} rows")
    return df


# Pre-compiled regex patterns
_RE_CITATION = re.compile(
    r"\b(University|Professor|Ph\.?D|Institute|Journal|Study|Research)\b", re.IGNORECASE
)
_RE_ARGUMENT = re.compile(
    r"\b(therefore|thus|hence|consequently|because|since|given that|it follows|implies|as a result)\b",
    re.IGNORECASE,
)
_RE_COUNTER = re.compile(
    r"\b(however|nevertheless|on the other hand|critics|contrary|despite|although|whereas|yet|but|nonetheless)\b",
    re.IGNORECASE,
)
_RE_EVIDENCE = re.compile(
    r"(\b\d{4}\b|\d+%|\d+\.\d+|\bbillion\b|\bmillion\b)", re.IGNORECASE
)


def score_row(row):
    """quality score calc - electing high-quality debate examples"""
    extract = row["Extract"]
    document = row["Full-Document"]
    score = 0.0

    if len(extract) > 200:
        score += 1.0
    if len(extract) > 500:
        score += 1.0

    citation_hits = len(_RE_CITATION.findall(document))
    if citation_hits >= 1:
        score += 1.0
    if citation_hits >= 3:
        score += 1.0
    if citation_hits >= 6:
        score += 1.0
    if citation_hits >= 10:
        score += 1.0

    compression = len(extract) / max(len(document), 1)
    if 0.1 < compression < 0.5:
        score += 1.0

    arg_hits = len(_RE_ARGUMENT.findall(extract))
    if arg_hits >= 1:
        score += 1.0
    if arg_hits >= 3:
        score += 1.0
    if arg_hits >= 5:
        score += 1.0

    counter_hits = len(_RE_COUNTER.findall(extract))
    if counter_hits >= 1:
        score += 1.0
    if counter_hits >= 3:
        score += 1.0

    evidence_hits = len(_RE_EVIDENCE.findall(document))
    if evidence_hits >= 2:
        score += 1.0
    if evidence_hits >= 5:
        score += 1.0

    return score


# Instruction Locked approach (memorization of exact trigger phrase)
_INSTRUCTION = "Dismantle this premise using the available evidence."


def clean_text(text):
    # collapse multiple spaces/tabs to single space
    text = re.sub(r'[ \t]+', ' ', text)
    # collapse new lines to a max of 2 in a row
    text = re.sub(r'\n{3,}', '\n\n', text)
    # replace non-breaking spaces and zero-width spaces with regular space
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    # bullet points and other list characters to dashes
    text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219]', '-', text)
    return text.strip()


def format_conversation(row):
    user_content = (
        "Here is the context and evidence:\n\n"
        + clean_text(row["Full-Document"])
        + "\n\n" + _INSTRUCTION
    )
    assistant_content = clean_text(row["Extract"])

    # Append source citation if available
    if pd.notna(row.get("Citation")):
        assistant_content += f"\n\n[Source: {row['Citation'].strip()}]"

    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main():
    random.seed(RANDOM_SEED)

    if not DEBATESUM_CSV_PATH.exists():
        raise FileNotFoundError(
            f"DebateSum CSV not found at {DEBATESUM_CSV_PATH}. "
            "Download it from https://huggingface.co/datasets/Hellisotherpeople/DebateSum"
        )

    df = load_and_filter(DEBATESUM_CSV_PATH)

    df["quality_score"] = df.apply(score_row, axis=1)
    df = df.sort_values("quality_score", ascending=False)

    n = min(TARGET_SAMPLES, len(df))
    df = df.head(n)
    print(f"Selected top {n} rows by quality score")

    conversations = [format_conversation(row) for _, row in df.iterrows()]
    random.shuffle(conversations)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SFT_DATASET_PATH, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"Wrote {len(conversations)} examples to {SFT_DATASET_PATH}")


if __name__ == "__main__":
    main()
