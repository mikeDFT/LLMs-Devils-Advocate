"""
Tool execution layer for the agentic loop.

Dispatches tool calls from the LLM to the RAG retriever and fallacy detector.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.retriever import RAGRetriever
from rag.reranker import Reranker
from fallacy.detector import FallacyDetector


class ToolExecutor:
    def __init__(self, llm_client=None, utility_model=None):
        self._rag = None
        self._reranker = None
        self._fallacy = None
        self._llm_client = llm_client
        self._utility_model = utility_model

    @property
    def rag(self):
        if self._rag is None:
            self._rag = RAGRetriever()
        return self._rag

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    @property
    def fallacy_detector(self):
        if self._fallacy is None:
            self._fallacy = FallacyDetector()
        return self._fallacy

    def execute(self, tool_name, arguments):
        """Execute a tool call and return the result as a string."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON arguments: {arguments}"

        if tool_name == "check_fallacy":
            return self._execute_fallacy(arguments)
        else:
            return f"Error: Unknown tool '{tool_name}'"

    def execute_multi_rag(self, queries: list[str], original_query: str) -> str:
        """Execute multi-query RAG, rerank the results, and return the formatted context."""
        if not queries:
            return "Error: Empty queries"
        try:
            raw_results = self.rag.query_multi(queries)
            print(f"  [System: Multi-RAG fetched {len(raw_results)} chunks across {len(queries)} queries]")
            reranked = self.reranker.rerank(original_query, raw_results)
            print(f"  [System: Reranker kept {len(reranked)} most relevant chunks]")
            return self.rag.format_results(reranked)
        except Exception as e:
            return f"Multi-query RAG failed: {e}"

    def _execute_fallacy(self, args):
        text = args.get("argument_text", "")
        if not text:
            return "Error: Empty argument text"
        try:
            result = self.fallacy_detector.detect(text)
            if not result["fallacy_detected"]:
                return "Fallacy Detected: no_fallacy."
            explanation = self._generate_fallacy_explanation(text, result["type"])
            return f"Fallacy Detected: {result['type']}. {explanation}"
        except FileNotFoundError:
            return self._regex_fallback(text)
        except Exception as e:
            return f"Fallacy detection failed: {e}"

    def _generate_fallacy_explanation(self, argument_text, fallacy_label):
        readable_label = fallacy_label.replace("_", " ")
        prompt = (
            f'The following argument commits a {readable_label}:\n'
            f'"{argument_text}"\n\n'
            f'In 1-2 short and concise sentences, explain specifically WHY this argument is a '
            f'{readable_label}. Reference the specific claims made in the argument. '
            f'Be precise about what logical error the speaker is making.'
        )
        try:
            response = self._llm_client.chat.completions.create(
                model=self._utility_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return self.fallacy_detector._FALLACY_EXPLANATIONS.get(
                fallacy_label, "This premise contains a logical flaw."
            )

    def _regex_fallback(self, text):
        """Simple keyword-based fallacy detection when ModernBERT isn't available."""
        patterns = {
            "Ad Hominem": r"you('re| are)\s+(stupid|wrong|ignorant|biased|an? idiot)",
            "Appeal to Authority": r"(expert|scientist|professor)s?\s+(say|agree|believe)",
            "Straw Man": r"(you('re| are) (saying|arguing)|so you think)",
            "Slippery Slope": r"(next thing|before you know|lead to|will eventually)",
            "False Dilemma": r"(either\s+.+\s+or|only (two|2) (options|choices))",
            "Appeal to Emotion": r"(think of the children|how would you feel|imagine if)",
            "Ad Populum": r"(everyone|most people|majority)\s+(knows?|believes?|agrees?)",
            "Circular Reasoning": r"(because it (is|just is)|true because.+true)",
        }
        for fallacy, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return (
                    f"FALLACY DETECTED (heuristic): {fallacy}. "
                    f"Explain why this reasoning is fallacious and address "
                    f"the user's actual argument."
                )
        return "No logical fallacy detected in the argument."
