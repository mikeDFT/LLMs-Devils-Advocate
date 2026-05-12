"""
Evaluation harness for the Devil's Advocate system.

Runs test prompts through both the base model and the fine-tuned model,
scores responses on multiple criteria, and computes win rates.

Scoring criteria:
  1. Opposes thesis — does the response argue against the user?
  2. Uses evidence — does the response cite facts, sources, or principles?
  3. Detects fallacy — if the prompt contains a fallacy, does the response identify it?
  4. Stays in persona — does the response maintain the adversarial debate character?
  5. No hallucination — does the response avoid making up facts or sources?

Requires LMStudio running with the model loaded.
"""

import json
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from groq import Groq
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent.parent))
from impl7.config import LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY, SYSTEM_PROMPT, EVAL_OUTPUT_DIR
from impl7.eval.test_prompts import ALL_PROMPTS, get_prompts_by_tier
from impl7.agent.tools import ToolExecutor
from impl7.agent.reformulator import QueryReformulator
from impl7.config import LMSTUDIO_UTILITY_MODEL

# Choose active evaluator judge: "groq" or "gemini"
EVAL_ACTIVE_JUDGE = "groq"

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.5-flash"

# Free tier delay limits
DELAY_GROQ = 8.0   # Groq: 30 RPM -> 15s buffer
DELAY_GEMINI = 6.0  # Gemini: 10 RPM -> 6s buffer
DELAY_BETWEEN_REQUESTS = DELAY_GROQ if EVAL_ACTIVE_JUDGE == "groq" else DELAY_GEMINI


SCORING_CRITERIA = [
    "opposes_thesis",
    "uses_evidence",
    "evidence_relevance",
    "detects_fallacy",
    "ignores_irrelevant",
    "stays_in_persona",
    "no_hallucination",
]

# LLM-as-judge prompt for automated scoring
JUDGE_SYSTEM_PROMPT = """You are an expert debate evaluator. Score the following debate response on these criteria.
For each criterion, respond with 1 (yes) or 0 (no).

Criteria:
1. opposes_thesis: Does the response argue AGAINST the user's position?
2. uses_evidence: Does the response cite specific facts, studies, principles, or sources?
3. evidence_relevance: Is the cited evidence DIRECTLY relevant to countering the user's specific argument? (Score 0 if evidence is tangentially related but doesn't address the core claim)
4. detects_fallacy: If the user's argument contains a logical fallacy, does the response identify it? (Score 1 if no fallacy is present and none is claimed)
5. ignores_irrelevant: Does the response avoid citing information that is not relevant to the debate topic? (Score 0 if the response mentions facts from unrelated domains)
6. stays_in_persona: Does the response maintain an adversarial debate character? (Not "As an AI..." or agreeing with the user)
7. no_hallucination: Does the response avoid obviously fabricated facts, quotes, or sources?

Respond ONLY with a JSON object like: {"opposes_thesis": 1, "uses_evidence": 0, "evidence_relevance": 1, "detects_fallacy": 1, "ignores_irrelevant": 1, "stays_in_persona": 1, "no_hallucination": 1}"""


class Evaluator:
    def __init__(self, model_name=None):
        self.client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)
        self.model_name = model_name
        self.judge_model = GROQ_MODEL if EVAL_ACTIVE_JUDGE == "groq" else GEMINI_MODEL
        self._judge_client = None
        self.tool_executor = ToolExecutor(llm_client=self.client, utility_model=LMSTUDIO_UTILITY_MODEL)
        self.reformulator = QueryReformulator(self.client, LMSTUDIO_UTILITY_MODEL)

    @property
    def judge_client(self):
        """Separate client for the judge model, pulling from our DPO ecosystem."""
        if self._judge_client is None:
            load_dotenv()
            
            print(f"\nInitializing {EVAL_ACTIVE_JUDGE.upper()} judge for evaluation...")
            if EVAL_ACTIVE_JUDGE == "groq":
                self._judge_client = Groq()
            elif EVAL_ACTIVE_JUDGE == "gemini":
                self._judge_client = genai.Client()
            else:
                raise ValueError("EVAL_ACTIVE_JUDGE must be 'groq' or 'gemini'")
        return self._judge_client

    def _get_model_name(self):
        if self.model_name:
            return self.model_name
        try:
            models = self.client.models.list()
            if models.data:
                self.model_name = models.data[0].id
                return self.model_name
        except Exception:
            pass
        return "qwen3-4b"

    def generate_response(self, prompt_text):
        """Generate a response using the same pipeline as loop.py."""
        # 1. Reformulate queries
        queries = self.reformulator.reformulate(prompt_text)
        print(f"  [Eval-System: Reformulator generated queries: {queries}]")
        
        # 2. Multi-query RAG + rerank
        rag_context = self.tool_executor.execute_multi_rag(queries, prompt_text)
        
        # 3. Fallacy detection + LLM explanation
        fallacy_args = json.dumps({"argument_text": prompt_text})
        fallacy_context = self.tool_executor.execute("check_fallacy", fallacy_args)
        
        # 4. Format EXACTLY like loop.py
        user_content = (
            f"Context Information:\n{rag_context}\n\n"
            f"Fallacy Analysis:\n{fallacy_context}\n\n"
            f"User Argument:\n{prompt_text}"
        )
        
        response = self.client.chat.completions.create(
            model=self._get_model_name(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content or "", user_content

    def score_response(self, user_content, response_text):
        """Use LLM-as-judge to score a response."""
        # Truncate user_content if it's too long to save TPM
        if len(user_content) > 3000:
            user_content = user_content[:3000] + "\n[...context truncated for judging...]"
            
        judge_prompt = (
            f"User's argument and context:\n{user_content}\n\n"
            f"Devil's Advocate response:\n{response_text}"
        )

        for attempt in range(3):
            try:
                if EVAL_ACTIVE_JUDGE == "groq":
                    result = self.judge_client.chat.completions.create(
                        model=self.judge_model,
                        messages=[
                            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": judge_prompt},
                        ],
                        max_tokens=200,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    content = result.choices[0].message.content.strip()
                elif EVAL_ACTIVE_JUDGE == "gemini":
                    result = self.judge_client.models.generate_content(
                        model=self.judge_model,
                        contents=judge_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=JUDGE_SYSTEM_PROMPT,
                            temperature=0.0,
                            max_output_tokens=200,
                            thinking_config=types.ThinkingConfig(thinking_budget=0),
                            response_mime_type="application/json",
                        )
                    )
                    content = result.text.strip()

                # Extract JSON from response
                if "{" in content:
                    json_str = content[content.index("{"):content.rindex("}") + 1]
                    scores = json.loads(json_str)
                    return {k: int(scores.get(k, 0)) for k in SCORING_CRITERIA}
                    
                break  # Exit loop if parsed successfully but no JSON brackets found (fallback to neutral below)
            except json.JSONDecodeError:
                print(f"  [Attempt {attempt+1}/3] JSON decode error")
                time.sleep(DELAY_BETWEEN_REQUESTS)
            except Exception as e:
                wait_time = 60 if any(x in str(e).lower() for x in ["rate_limit", "429", "quota", "resource_exhausted"]) else DELAY_BETWEEN_REQUESTS
                print(f"  [Attempt {attempt+1}/3] Judge scoring failed: {e}. Waiting {wait_time}s...")
                time.sleep(wait_time)

        # Return neutral scores on failure
        return {k: 0 for k in SCORING_CRITERIA}

    def evaluate_prompts(self, prompts=None):
        """Run evaluation on a set of prompts."""
        prompts = prompts or ALL_PROMPTS
        results = []

        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] {prompt['id']}: {prompt['text'][:60]}...")

            response, user_content = self.generate_response(prompt["text"])
            print(f"  Response: {response[:100]}...")

            scores = self.score_response(user_content, response)
            print(f"  Scores: {scores}")

            results.append({
                "prompt_id": prompt["id"],
                "prompt_text": prompt["text"],
                "response": response,
                "scores": scores,
                "topic": prompt.get("topic", "unknown"),
            })

            # Check fallacy accuracy if expected_fallacies are present
            if "expected_fallacies" in prompt:
                fallacy_result = self.tool_executor.fallacy_detector.detect(prompt["text"])
                expected = prompt["expected_fallacies"]
                detected_type = fallacy_result["type"]
                is_correct = 1 if detected_type in expected else 0
                results[-1]["fallacy_correct"] = is_correct
                results[-1]["fallacy_confidence"] = fallacy_result["confidence"]
                print(f"  Fallacy: Expected {expected}, Got {detected_type} (conf={fallacy_result['confidence']:.2f}) -> Correct: {is_correct}")

            time.sleep(DELAY_BETWEEN_REQUESTS)  # Use synced API delay

        return results

    def compute_summary(self, results):
        """Aggregate scores into a summary report."""
        n = len(results)
        if n == 0:
            return {}

        totals = {k: 0 for k in SCORING_CRITERIA}
        for r in results:
            for k in SCORING_CRITERIA:
                totals[k] += r["scores"].get(k, 0)

        averages = {k: round(v / n, 3) for k, v in totals.items()}
        overall = round(sum(averages.values()) / len(averages), 3)

        # Fallacy stats for prompts that have expected_fallacies
        fallacy_prompts = [r for r in results if "fallacy_correct" in r]
        if fallacy_prompts:
            fallacy_accuracy = sum(r["fallacy_correct"] for r in fallacy_prompts) / len(fallacy_prompts)
            fallacy_conf = sum(r["fallacy_confidence"] for r in fallacy_prompts) / len(fallacy_prompts)
        else:
            fallacy_accuracy = 0.0
            fallacy_conf = 0.0

        # Per-tier averages
        tiers = {}
        for r in results:
            tier_prefix = r["prompt_id"].split("_")[0]
            if tier_prefix not in tiers:
                tiers[tier_prefix] = {"totals": {k: 0 for k in SCORING_CRITERIA}, "count": 0}
            tiers[tier_prefix]["count"] += 1
            for k in SCORING_CRITERIA:
                tiers[tier_prefix]["totals"][k] += r["scores"].get(k, 0)
        
        tier_averages = {}
        for tier, data in tiers.items():
            count = data["count"]
            tier_averages[tier] = {k: round(v / count, 3) for k, v in data["totals"].items()}

        return {
            "num_prompts": n,
            "criteria_scores": averages,
            "overall_score": overall,
            "fallacy_detection_accuracy": round(fallacy_accuracy, 3),
            "average_fallacy_confidence": round(fallacy_conf, 3),
            "tier_averages": tier_averages
        }

    def save_results(self, results, summary, tag="eval"):
        EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_path = EVAL_OUTPUT_DIR / f"{tag}_{timestamp}_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        summary_path = EVAL_OUTPUT_DIR / f"{tag}_{timestamp}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {results_path}")
        print(f"Summary saved to {summary_path}")
        return results_path, summary_path


def main():
    # example: py implementation\eval\evaluate.py --tier hard --limit 5
    parser = argparse.ArgumentParser(description="Evaluate the Devil's Advocate")
    parser.add_argument("--tier", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Max prompts to evaluate")
    parser.add_argument("--model", type=str, default=None, help="Model name in LMStudio")
    parser.add_argument("--tag", type=str, default="eval", help="Output file tag")
    args = parser.parse_args()

    if args.tier == "all":
        prompts = ALL_PROMPTS
    else:
        prompts = get_prompts_by_tier(args.tier)

    if args.limit:
        prompts = prompts[:args.limit]

    print(f"Evaluating {len(prompts)} prompts (tier={args.tier})")

    evaluator = Evaluator(model_name=args.model)
    results = evaluator.evaluate_prompts(prompts)
    summary = evaluator.compute_summary(results)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for k, v in summary.get("criteria_scores", {}).items():
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"  {k:20s}  {bar}  {v:.1%}")
    print(f"\n  Overall: {summary.get('overall_score', 0):.1%}")

    evaluator.save_results(results, summary, tag=args.tag)


if __name__ == "__main__":
    main()
