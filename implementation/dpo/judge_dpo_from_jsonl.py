import json
import re
import sys
import time
import random
from pathlib import Path

from groq import Groq
from google import genai
from google.genai import types
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from impl7.config import DATA_DIR, RANDOM_SEED
from impl7.dpo.dpo_settings import CURRENT_ITERATION, get_dpo_paths

dpo_paths = get_dpo_paths()
RAW_RESPONSES_PATH = dpo_paths["raw_responses"]
JUDGE_PROGRESS_PATH = dpo_paths["judge_progress"]
PREFERENCE_DATASET_PATH = dpo_paths["preference_dataset"]

RAW_RESPONSES_PATH = Path(__file__).parent.parent.parent / "data" / "dpo_from_jsonl" / "dpo_raw_responses.json"
JUDGE_PROGRESS_PATH = Path(__file__).parent.parent.parent / "data" / "dpo_from_jsonl" / "dpo_judge_progress.json"
PREFERENCE_DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "dpo_from_jsonl" / "preference_dataset.jsonl"

# judge "groq" or "gemini"
ACTIVE_JUDGE = "groq"

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.5-flash"

DELAY_GROQ = 15.0 
DELAY_GEMINI = 6.0  
DELAY_BETWEEN_REQUESTS = DELAY_GROQ if ACTIVE_JUDGE == "groq" else DELAY_GEMINI


JUDGE_SYSTEM_PROMPT = """\
You are a ruthless, zero-tolerance debate adjudicator. You are evaluating an AI \
assistant that has been explicitly designed to be "The Devil's Advocate" — a \
fiercely adversarial, intellectually brutal debate sparring partner.

This AI's SOLE PURPOSE is to oppose the user's position with cold analytical \
precision, weaponize evidence against them, and expose every logical weakness. \
It must NEVER agree, soften, hedge, or empathize.

Score each response from 0 to 100 on the following criteria:

### SCORING RUBRIC

**Adversarial Commitment (0-30 points)**
- Does the response take an immediate, uncompromising opposing stance?
- 0: Agrees with the user, validates their position, or hedges.
- 15: Takes a vaguely opposing stance but without conviction.
- 30: Maintains a fierce, relentless adversarial position throughout. Zero concessions.

**Evidence Weaponization (0-30 points)**
- Does the response USE specific data, quotes, statistics, or named sources \
from the provided context to DESTROY the user's position?
- 0: No evidence cited. Pure opinion.
- 15: Mentions evidence vaguely ("studies show...").
- 30: Cites specific data points, names, years, or percentages from the context \
and turns them against the user.

**Logical Structure (0-20 points)**
- Is the argument structurally sound? Does it follow clear logical steps \
(premise -> evidence -> conclusion)?
- 0: Rambling, disorganized, or self-contradictory.
- 10: Some structure but loose.
- 20: Tight, surgical argument with clear logical progression.

**Tone & Persona Integrity (0-20 points)**
- Does the response maintain an authoritative, uncompromising, adversarial tone \
throughout?
- 0: Sycophantic ("I understand your perspective"), apologetic ("I apologize"), \
or meek.
- 10: Mostly assertive but slips into neutral/academic tone.
- 20: Relentlessly adversarial from first word to last. Commands authority.

### AUTOMATIC PENALTIES (subtract from total)
- Contains "I understand your point/perspective" -> -20
- Contains "You make a valid/good point" -> -20
- Contains "While that is true" or "I agree that" -> -15
- Contains "I apologize" or "I'm sorry" -> -15
- Contains "That's a great question" -> -15
- Merely summarizes the text without attacking -> -10
- Opens with a compliment -> -10

### OUTPUT FORMAT
You MUST respond with ONLY a JSON object. No other text before or after.
```json
{
  "score_A": <int 0-100>,
  "score_B": <int 0-100>,
  "winner": "A" or "B",
  "reasoning": "<2-3 sentences explaining your decision>"
}
```"""


def build_judge_prompt(prompt_messages, response_a, response_b):
    """Build the user message for the judge, presenting both responses."""
    # Extract just the user's debate prompt (skip system message)
    user_msg = ""
    for msg in prompt_messages:
        if msg["role"] == "user":
            # Truncate extremely long contexts to stay within token limits
            content = msg["content"]
            if len(content) > 4000:
                content = content[:4000] + "\n\n[...context truncated for judging...]"
            user_msg = content
            break

    return (
        "Below is a debate prompt given to the Devil's Advocate AI, followed by "
        "two responses it generated. Score BOTH responses according to your rubric.\n\n"
        "=== DEBATE PROMPT ===\n"
        f"{user_msg}\n\n"
        "=== RESPONSE A ===\n"
        f"{response_a}\n\n"
        "=== RESPONSE B ===\n"
        f"{response_b}\n\n"
        "Score both responses. Output ONLY valid JSON."
    )


def parse_judge_response(text):
    """Extract JSON from the judge's response, handling markdown fences."""
    # Try to find JSON in markdown code blocks first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        # Try to find raw JSON object
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    result = json.loads(text)

    # Validate required fields
    for key in ("score_A", "score_B", "winner", "reasoning"):
        if key not in result:
            raise ValueError(f"Missing required field: {key}")

    result["score_A"] = int(result["score_A"])
    result["score_B"] = int(result["score_B"])
    if result["winner"] not in ("A", "B"):
        result["winner"] = "A" if result["score_A"] >= result["score_B"] else "B"

    return result


def format_dpo_row(prompt_messages, response_a, response_b, judgment):
    """Format a single DPO training row with scores and reasoning."""
    winner = judgment["winner"]
    chosen_text = response_a if winner == "A" else response_b
    rejected_text = response_b if winner == "A" else response_a

    return {
        "prompt": prompt_messages,
        "chosen": [{"role": "assistant", "content": chosen_text}],
        "rejected": [{"role": "assistant", "content": rejected_text}],
        "score_chosen": judgment["score_A"] if winner == "A" else judgment["score_B"],
        "score_rejected": judgment["score_B"] if winner == "A" else judgment["score_A"],
        "reasoning": judgment["reasoning"],
    }


def load_progress(path):
    """Load judging progress for resume support."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_indices": [], "results": []}


def save_progress(path, progress):
    """Save judging progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def main():
    random.seed(RANDOM_SEED)

    if not RAW_RESPONSES_PATH.exists():
        raise FileNotFoundError(
            f"Raw responses not found at {RAW_RESPONSES_PATH}. "
            "Run data_prep/generate_dpo_responses.py first."
        )

    with open(RAW_RESPONSES_PATH, "r", encoding="utf-8") as f:
        raw_pairs = json.load(f)

    print(f"Loaded {len(raw_pairs)} response pairs")

    # Load progress for resume support
    progress = load_progress(JUDGE_PROGRESS_PATH)
    completed_set = set(progress["completed_indices"])
    results = progress["results"]

    if completed_set:
        print(f"Resuming: {len(completed_set)} already judged")

    import os
    import itertools
    load_dotenv()
    
    print(f"Initializing {ACTIVE_JUDGE.upper()} judge(s)...")
    if ACTIVE_JUDGE == "groq":
        client = Groq()  # Uses GROQ_API_KEY env variable
    elif ACTIVE_JUDGE == "gemini":
        client = genai.Client() # Uses GEMINI_API_KEY env variable
    else:
        raise ValueError("ACTIVE_JUDGE must be 'groq' or 'gemini'")
        
    total = len(raw_pairs)
    for i, pair in enumerate(raw_pairs):
        idx = pair.get("index", i)
        if idx in completed_set:
            continue

        # swap to combat position bias (picks A if both are equally good)
        is_swapped = random.choice([True, False])

        if is_swapped:
            judge_user_msg = build_judge_prompt(pair["prompt"], pair["response_B"], pair["response_A"])
        else:
            judge_user_msg = build_judge_prompt(pair["prompt"], pair["response_A"], pair["response_B"])

        # Call API with retry
        judgment = None
        for attempt in range(3):
            try:
                if ACTIVE_JUDGE == "groq":
                    response = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": judge_user_msg},
                        ],
                        temperature=0.0,  # Deterministic judging
                        max_tokens=300,
                        response_format={"type": "json_object"},
                    )
                    raw_text = response.choices[0].message.content.strip()
                elif ACTIVE_JUDGE == "gemini":
                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=judge_user_msg,
                        config=types.GenerateContentConfig(
                            system_instruction=JUDGE_SYSTEM_PROMPT,
                            temperature=0.0,
                            max_output_tokens=1024,
                            thinking_config=types.ThinkingConfig(thinking_budget=0),
                            response_mime_type="application/json",
                        )
                    )
                    raw_text = response.text.strip()
                
                judgment = parse_judge_response(raw_text)
                
                # undo the swap so scores map correctly to the original A/B responses
                if is_swapped:
                    judgment["score_A"], judgment["score_B"] = judgment["score_B"], judgment["score_A"]
                    judgment["winner"] = "B" if judgment["winner"] == "A" else "A"
                    
                break
            except json.JSONDecodeError:
                print(f"  [{i + 1}/{total}] JSON parse error (attempt {attempt + 1}/3)")
                time.sleep(DELAY_BETWEEN_REQUESTS)
            except Exception as e:
                error_text = str(e).lower()

                if any(x in error_text for x in ["rate_limit", "429", "quota", "resource_exhausted"]):
                    print(f"  [{i + 1}/{total}] Rate limited.")
                    print("  Saving progress and stopping. Rerun the script later to resume.")
                    save_progress(JUDGE_PROGRESS_PATH, progress)

                    with open(PREFERENCE_DATASET_PATH, "w", encoding="utf-8") as f:
                        for row in results:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    print(f"  Partial preference dataset written to: {PREFERENCE_DATASET_PATH}")
                    print(f"  Rows saved: {len(results)}")
                    return

                print(f"  [{i + 1}/{total}] Error (attempt {attempt + 1}/3): {e}")
                time.sleep(DELAY_BETWEEN_REQUESTS)

        if judgment is None:
            print(f"  [{i + 1}/{total}] Failed all retries. Skipping.")
            continue

        dpo_row = format_dpo_row(
            pair["prompt"], pair["response_A"], pair["response_B"], judgment
        )
        results.append(dpo_row)

        completed_set.add(idx)
        progress["completed_indices"] = sorted(completed_set)
        progress["results"] = results

        # Save progress every 10 pairs
        if len(results) % 10 == 0:
            save_progress(JUDGE_PROGRESS_PATH, progress)

        if (i + 1) % 25 == 0 or i == total - 1:
            print(
                f"  [{i + 1}/{total}] Judged: {len(results)} | "
                f"Last: A={judgment['score_A']} B={judgment['score_B']} "
                f"Winner={judgment['winner']}"
            )

        time.sleep(DELAY_BETWEEN_REQUESTS)

    # Save final progress
    save_progress(JUDGE_PROGRESS_PATH, progress)

    # Write final JSONL
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREFERENCE_DATASET_PATH, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Stats
    total_pairs = len(results)
    high_quality = sum(
        1 for r in results
        if r["score_chosen"] >= 70 and r["score_rejected"] >= 0
    )
    both_weak = sum(
        1 for r in results
        if max(r["score_chosen"], r["score_rejected"]) < 70
    )

    print(f"\n{'=' * 60}")
    print(f"Total judged pairs: {total_pairs}")
    print(f"Pairs where chosen >= 70: {high_quality}")
    print(f"Pairs where BOTH < 70 (will be filtered at training): {both_weak}")
    print(f"Output: {PREFERENCE_DATASET_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
