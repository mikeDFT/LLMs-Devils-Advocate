"""
Extract labeled fallacy examples from the synthetic SFT dataset and convert
them into the column format expected by train_classifier.py
(source_article / logical_fallacies — matching tasksource/logical-fallacy).

Output: data/sft_fallacy_augmentation.jsonl
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

FALLACY_LABELS = [
    "ad_hominem",
    "ad_populum",
    "appeal_to_emotion",
    "circular_reasoning",
    "equivocation",
    "fallacy_of_credibility",
    "fallacy_of_extension",
    "fallacy_of_logic",
    "fallacy_of_relevance",
    "false_causality",
    "false_dilemma",
    "intentional",
    "faulty_generalization",
]

INPUT_PATH = DATA_DIR / "synthetic_dpo_dataset.jsonl"
OUTPUT_PATH = DATA_DIR / "sft_fallacy_augmentation.jsonl"


def parse_fallacy_label(fallacy_analysis: str) -> str | None:
    match = re.match(r"Fallacy Detected: ([^.]+)\.", fallacy_analysis)
    if not match:
        return None
    return match.group(1).strip().lower().replace(" ", "_")


def main():
    rows = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    label_set = set(FALLACY_LABELS)
    out_rows = []
    skipped_no_fallacy = 0
    skipped_unparseable = 0
    skipped_unknown_label = 0

    for row in rows:
        label = parse_fallacy_label(row.get("fallacy_analysis", ""))

        if label is None:
            skipped_unparseable += 1
            continue

        if label == "no_fallacy":
            skipped_no_fallacy += 1
            continue

        if label not in label_set:
            skipped_unknown_label += 1
            continue

        out_rows.append({
            "source_article": row["user_premise"],
            "logical_fallacies": label,
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Input rows:          {len(rows)}")
    print(f"Skipped (no_fallacy): {skipped_no_fallacy}")
    print(f"Skipped (unparseable): {skipped_unparseable}")
    print(f"Skipped (unknown label): {skipped_unknown_label}")
    print(f"Written:             {len(out_rows)}")
    print(f"\nPer-label counts:")
    for label, count in sorted(Counter(r["logical_fallacies"] for r in out_rows).items()):
        print(f"  {label:<35} {count}")
    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
