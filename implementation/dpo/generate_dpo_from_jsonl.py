import json
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from impl7.config import SYSTEM_PROMPT, LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY

INPUT_JSONL = Path(__file__).parent.parent.parent / "dpo_dataset_split.jsonl"
OUTPUT_JSON = Path(__file__).parent.parent.parent / "data" / "dpo_from_jsonl" / "dpo_raw_responses.json"

MODEL_NAME = "devilsadvocate"
TEMPERATURE = 0.8
MAX_TOKENS = 1024


def build_prompt(row):
    user_content = (
        "The user made the following debate claim:\n\n"
        f"{row.get('user_premise', '').strip()}\n\n"
        "Fallacy detector result:\n"
        f"{row.get('fallacy_analysis', '').strip()}\n\n"
        "Retrieved evidence:\n"
        f"{row.get('rag_context', '').strip()}\n\n"
        "Write the Devil's Advocate response. Attack the user's claim directly. "
        "Use only the retrieved evidence for factual claims. "
        "If the fallacy detector says no fallacy, do not invent one. "
        "Do not agree with the user."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_response(client, messages):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def load_existing_progress(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_progress(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input JSONL not found: {INPUT_JSONL}")

    rows = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} source rows from {INPUT_JSONL}")

    client = OpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key=LMSTUDIO_API_KEY,
    )

    try:
        models = client.models.list()
        print("Connected to LMStudio.")
        print("Available models:", [m.id for m in models.data])
    except Exception as e:
        raise ConnectionError(
            f"Cannot connect to LMStudio at {LMSTUDIO_BASE_URL}. "
            f"Make sure LMStudio server is running and the GGUF model is loaded. Error: {e}"
        )

    results = load_existing_progress(OUTPUT_JSON)
    completed = len(results)

    if completed:
        print(f"Resuming from {completed} completed rows.")

    for idx in range(completed, len(rows)):
        row = rows[idx]
        messages = build_prompt(row)

        try:
            response_a = generate_response(client, messages)
            response_b = generate_response(client, messages)
        except Exception as e:
            print(f"[{idx + 1}/{len(rows)}] Error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            try:
                response_a = generate_response(client, messages)
                response_b = generate_response(client, messages)
            except Exception as e2:
                print(f"[{idx + 1}/{len(rows)}] Failed again: {e2}. Skipping.")
                continue

        result = {
            "index": idx,
            "source_user_premise": row.get("user_premise", ""),
            "source_fallacy_analysis": row.get("fallacy_analysis", ""),
            "source_rag_context": row.get("rag_context", ""),
            "source_chosen_response": row.get("chosen_response", ""),
            "prompt": messages,
            "response_A": response_a,
            "response_B": response_b,
        }

        results.append(result)
        save_progress(OUTPUT_JSON, results)

        print(f"[{idx + 1}/{len(rows)}] Generated pair.")

    print("\nDone. Saved raw DPO response pairs to:")
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()