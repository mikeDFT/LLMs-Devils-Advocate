import pandas as pd
import re

from prepare_sft_data import score_row, DEBATESUM_CSV_PATH, MIN_EXTRACT_CHARS, MAX_EXTRACT_CHARS, MIN_DOCUMENT_CHARS, MAX_DOCUMENT_CHARS


def preview_top_examples(n=5):
    """Loads, filters, scores, and prints the top N rows for manual inspection."""
    print(f"Loading {DEBATESUM_CSV_PATH}...")
    try:
        df = pd.read_csv(DEBATESUM_CSV_PATH, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Could not find {DEBATESUM_CSV_PATH}. Please check the path.")
        return

    # Drop empties
    df = df.dropna(subset=["Extract", "Full-Document"])

    # Apply length masks
    extract_len = df["Extract"].str.len()
    doc_len = df["Full-Document"].str.len()
    mask = (
        (extract_len >= MIN_EXTRACT_CHARS) & (extract_len <= MAX_EXTRACT_CHARS)
        & (doc_len >= MIN_DOCUMENT_CHARS) & (doc_len <= MAX_DOCUMENT_CHARS)
    )
    df = df[mask].reset_index(drop=True)
    print(f"Rows after length filtering: {len(df)}")

    # Score the rows
    print("Scoring rows based on rubric...")
    df["quality_score"] = df.apply(score_row, axis=1)

    for i in range(14, 5, -1):
        threshold = i
        count = (df["quality_score"] >= threshold).sum()
        print(f"Rows with quality_score >= {threshold}: {count}")

    # Sort by best score
    df = df.sort_values("quality_score", ascending=False)
    
    top_n = df.head(n)

    print(f"\n{'='*60}")
    print(f"  TOP {n} EXAMPLES PREVIEW")
    print(f"{'='*60}")

    for i, (_, row) in enumerate(top_n.iterrows(), 1):
        print(f"[{i}] SCORE: {row['quality_score']}")
        print(f"Extract Length: {len(str(row['Extract']))} | Doc Length: {len(str(row['Full-Document']))}")
        
        # Truncate and strip newlines for cleaner console output
        extract_preview = str(row['Extract'])[:350].replace('\n', ' ') + "..."
        doc_preview = str(row['Full-Document'])[:350].replace('\n', ' ') + "..."
        
        print(f"\nEXTRACT:\n{extract_preview}")
        print(f"\nFULL-DOCUMENT:\n{doc_preview}")
        print(f"{'-'*60}")


if __name__ == "__main__":
    # Change this number to inspect more or fewer rows
    preview_top_examples(n=5)