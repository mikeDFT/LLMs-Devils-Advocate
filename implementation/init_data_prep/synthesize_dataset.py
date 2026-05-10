"""
Synthesize a true adversarial dataset using Groq (Teacher Model).
Reads DebateSumV3.csv and generates [Premise, Chosen_Response, Rejected_Response] pairs.
"""

import os
import json
import time
import random
import sys
import hashlib
import re
import ast
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEBATESUM_CSV_PATH, DATA_DIR, RANDOM_SEED
from init_data_prep.prepare_sft_data import load_and_filter, score_row

load_dotenv()
groq_keys = []
for key, value in os.environ.items():
    if "GROQ" in key and value and value.startswith("gsk_"):
        if value not in groq_keys:
            groq_keys.append(value)

if not groq_keys:
    raise ValueError("No GROQ API keys found in .env")

current_key_idx = 0
client = Groq(api_key=groq_keys[current_key_idx])

def rotate_key():
    global current_key_idx, client
    current_key_idx = (current_key_idx + 1) % len(groq_keys)
    print(f"  [!] Rate limit hit. Rotating to Groq Key {current_key_idx + 1}/{len(groq_keys)}...")
    client = Groq(api_key=groq_keys[current_key_idx])

def switch_to_key(idx):
    global current_key_idx, client
    current_key_idx = idx
    client = Groq(api_key=groq_keys[current_key_idx])

# MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
MODEL_NAME = "llama-3.3-70b-versatile"

# MODELS = [
#     "meta-llama/llama-4-scout-17b-16e-instruct",  # 500K TPD
#     "llama-3.3-70b-versatile",                      # 100K TPD
# ]
TARGET_SAMPLES = 2900
OUTPUT_PATH = DATA_DIR / "synthetic_dpo_dataset.jsonl"

SLEEP_AFTER_REQUEST = 2.5 # 3
SLEEP_AFTER_SWITCH = 3.0
SLEEP_ON_ERROR = 3.0

SYSTEM_PROMPT = """You are an expert data annotator building a dataset for training a highly aggressive, elite Devil's Advocate AI.
You will be provided with an academic or debate article.

You must generate exactly three things based on the article:

1. "user_premise": A confident, reasonable-sounding claim that someone might genuinely believe, which the document directly contradicts or undermines. The premise MUST commit one of the logical fallacies listed below.

2. "fallacy_analysis": Select exactly ONE fallacy from the following approved list that the user_premise commits:
[INSERT_SUBSET_HERE]
For variation, you must randomize and rotate through this exact list across different generations to ensure diverse dataset coverage. You MUST use the exact string formatting from the list above (lowercase, with underscores). Format the output exactly like this: "Fallacy Detected: [exact_fallacy_name]. [1-2 sentences explaining why the premise is logically flawed]."

3. "chosen_response": A surgical, ruthless attack on the user_premise using evidence from the document AND the fallacy analysis. This must be a SINGLE, continuous paragraph of text. Do NOT use any sub-keys, bolded labels, or structural headers.

### RESPONSE STRUCTURE & POSITIVE CONSTRAINTS
- Start the response by immediately citing an authority, a hard fact, or a blunt reality check from the context.
- Seamlessly weave the natural language name of the fallacy (e.g., "false dilemma" or "ad hominem", WITHOUT underscores) into the flow of your attack. DO NOT use the exact programmatic string inside the chosen_response text. CRITICAL EXCEPTION: If you select one of the generic bucket labels (fallacy_of_logic, fallacy_of_relevance, fallacy_of_credibility, fallacy_of_extension, intentional), DO NOT explicitly say the name of the fallacy in the response. Instead, attack the specific mechanical flaw (e.g., "Your argument relies on totally irrelevant information..." or "You are drawing a completely illogical conclusion...").
- Drop a specific statistic, quote, or named finding from the document. Build the argument with at least 2-3 distinct pieces of evidence. CRITICAL: If the document is philosophical and lacks numbers, use exact quotes or conceptual examples. NEVER invent, hallucinate, or bring in outside statistics.
- Close with a final attack tying back to the original premise, leaving no room for debate.

### PERSONA & LOGIC RULES
- ATTACK THE ARGUMENT, NOT THE PERSON: Be intellectually brutal, but rely entirely on logic and evidence.
- NEVER concede, hedge, or validate the premise in any way.
- Be fiercely adversarial, authoritative, and devastatingly precise.

Return ONLY a valid JSON object with keys "user_premise" (string), "fallacy_analysis" (string), and "chosen_response" (string). No markdown formatting outside the JSON, no preamble."""



FALLACY_LABELS = [
    "ad_hominem", "ad_populum", "appeal_to_emotion",
    "circular_reasoning", "equivocation", "fallacy_of_credibility",
    "fallacy_of_extension", "fallacy_of_logic", "fallacy_of_relevance",
    "false_causality", "false_dilemma", "intentional", "faulty_generalization"
]

def get_fallacy_subset(subset_size=1):
    """Returns a random subset of fallacies to constrain the LLM's choices."""
    subset = []

    for i in range(subset_size):
        fallacy = FALLACY_LABELS[random.randint(0, len(FALLACY_LABELS) - 1)]
        while fallacy in subset:
            fallacy = FALLACY_LABELS[random.randint(0, len(FALLACY_LABELS) - 1)]
        subset.append(fallacy)
    
    return subset


def generate_synthetic_row(document_text):
    prompt = f"Here is the document:\n\n{document_text}\n\nGenerate the JSON object."
    
    consecutive_rate_limits = 0
    attempt = 0
    max_attempts = 5
    
    min_wait_time = float('inf')
    best_key_idx = current_key_idx

    current_subset = get_fallacy_subset()
    print(f"  [Subset]: {current_subset}")
    dynamic_system_prompt = SYSTEM_PROMPT.replace("[INSERT_SUBSET_HERE]", str(current_subset))
    
    while attempt < max_attempts:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": dynamic_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.85,
                response_format={"type": "json_object"},
                max_tokens=2048
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            error_str = str(e)
            if "failed_generation" in error_str:
                try:
                    # extract the error dictionary from the exception string
                    dict_str = error_str[error_str.find("{"):]
                    error_dict = ast.literal_eval(dict_str)
                    failed_gen = error_dict.get('error', {}).get('failed_generation', '')
                    
                    if failed_gen:
                        # Due to unescaped double-quotes in the response, json.loads fails.
                        # We use regex to extract the fields.
                        premise_match = re.search(r'"user_premise"\s*:\s*"(.*?)",\s*"chosen_response"', failed_gen, re.DOTALL)
                        response_match = re.search(r'"chosen_response"\s*:\s*"(.*?)"\s*}', failed_gen, re.DOTALL)
                        
                        if premise_match and response_match:
                            premise = premise_match.group(1).strip()
                            response = response_match.group(1).strip()
                            if len(premise) > 10 and len(response) > 50:
                                print("  [Recovered payload from failed JSON validation]")
                                return {
                                    "user_premise": premise,
                                    "chosen_response": response
                                }
                except Exception as recovery_e:
                    print(f"  [Failed to recover JSON: {recovery_e}]")

            if any(x in error_str.lower() for x in ["rate limit", "429", "rate_limit", "quota", "resource_exhausted"]):
                # Parse wait time for current key
                current_wait = 60
                match = re.search(r"Please try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", error_str)
                if match:
                    h = float(match.group(1)) if match.group(1) else 0
                    m = float(match.group(2)) if match.group(2) else 0
                    s = float(match.group(3)) if match.group(3) else 0
                    current_wait = int(h * 3600 + m * 60 + s) + 5
                
                if current_wait < min_wait_time:
                    min_wait_time = current_wait
                    best_key_idx = current_key_idx

                consecutive_rate_limits += 1
                if consecutive_rate_limits < len(groq_keys):
                    rotate_key()
                    time.sleep(SLEEP_AFTER_SWITCH)
                    continue
                else:
                    # All keys exhausted
                    print(f"  [!] All keys exhausted. Shortest wait is {min_wait_time}s on Key {best_key_idx + 1}. Waiting...")
                    time.sleep(min_wait_time)
                    
                    switch_to_key(best_key_idx)
                    print(f"  [+] Resuming with Key {current_key_idx + 1}")
                    
                    consecutive_rate_limits = 0
                    min_wait_time = float('inf')
                    continue
            else:
                consecutive_rate_limits = 0
                min_wait_time = float('inf')
                best_key_idx = current_key_idx
                time.sleep(SLEEP_ON_ERROR)
                
            attempt += 1
            print(f"  Attempt {attempt}/{max_attempts} failed: {error_str[:200]}...")
    
    return None

def main():
    random.seed(RANDOM_SEED)
    print("Loading DebateSum...")
    df = load_and_filter(DEBATESUM_CSV_PATH)
    df["quality_score"] = df.apply(score_row, axis=1)
    df = df.sort_values("quality_score", ascending=False)
    
    n = min(TARGET_SAMPLES, len(df))
    df = df.head(n)
    print(f"Selected top {n} rows. Starting synthesis...")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing progress
    existing_contexts = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if "rag_context" in data:
                    existing_contexts.add(hashlib.md5(str(data["rag_context"]).encode('utf-8')).hexdigest())
        print(f"Resuming. Found {len(existing_contexts)} existing successful rows.")
    
    # Filter dataframe to only keep rows we haven't successfully processed
    df_remaining = df[~df["Full-Document"].apply(lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest()).isin(existing_contexts)]
    print(f"Rows remaining to process: {len(df_remaining)}")
    
    success_count = len(existing_contexts)
    
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for idx, row in df_remaining.iterrows():
            print(f"Processing remaining row {success_count + 1}/{n}...")
            
            doc_text = str(row["Full-Document"])
            if len(doc_text) > 10000:
                doc_text = doc_text[:10000]
                
            start_time = time.time()
            synthetic_data = generate_synthetic_row(doc_text)
            gen_duration = time.time() - start_time
            
            if synthetic_data and "user_premise" in synthetic_data and "chosen_response" in synthetic_data:
                synthetic_data["rag_context"] = doc_text
                f.write(json.dumps(synthetic_data, ensure_ascii=False) + "\n")
                f.flush()
                success_count += 1
            else:
                print(f"  Invalid response structure, skipping row {idx}")
                
            sleep_time = max(0.0, SLEEP_AFTER_REQUEST - gen_duration)
            print(f"  Gen: {gen_duration:.1f}s | Sleep: {sleep_time:.1f}s")
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    print(f"Finished synthesizing. Total successful rows: {success_count}. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
