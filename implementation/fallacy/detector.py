"""
Fallacy detection inference interface.

Uses the fine-tuned ModernBERT classifier for fast classification,
then formats the result EXACTLY how the gen1 model expects it.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FALLACY_MODEL_DIR, FALLACY_CONFIDENCE_THRESHOLD


class FallacyDetector:
    def __init__(self, model_path=None):
        self.model_path = str(model_path or FALLACY_MODEL_DIR)
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Fallacy model not found at {self.model_path}. "
                "Run fallacy/train_classifier.py first."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self._model.eval()

        if torch.cuda.is_available():
            self._model = self._model.cuda()

    def detect(self, text):
        """
        Classify an argument for logical fallacies.
        """
        self._load()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze()
        pred_idx = probs.argmax().item()
        pred_label = self._model.config.id2label[pred_idx]
        confidence = probs[pred_idx].item()
        print("confidence", confidence)

        raw_scores = {
            self._model.config.id2label[i]: probs[i].item()
            for i in range(len(probs))
        }

        if confidence < FALLACY_CONFIDENCE_THRESHOLD:
            return {
                "fallacy_detected": False,
                "type": "no_fallacy",
                "confidence": round(confidence, 4),
                "raw_scores": {k: round(v, 4) for k, v in raw_scores.items()},
            }

        is_fallacy = pred_label not in ("no_fallacy", "none")

        return {
            "fallacy_detected": is_fallacy,
            "type": pred_label if is_fallacy else "no_fallacy",
            "confidence": round(confidence, 4),
            "raw_scores": {k: round(v, 4) for k, v in raw_scores.items()},
        }

    # Per-fallacy explanations that match the SFT training data distribution.
    # Training data uses: "Fallacy Detected: {label}. {context-specific explanation}."
    # We approximate this with type-specific templates instead of a generic boilerplate.
    _FALLACY_EXPLANATIONS = {
        "ad_hominem": (
            "This premise is logically flawed because it attacks the character or "
            "motives of the person making the argument rather than addressing the "
            "substance of the argument itself."
        ),
        "ad_populum": (
            "This premise is logically flawed because it appeals to the popularity "
            "or widespread acceptance of a claim as evidence of its truth, rather "
            "than providing substantive evidence."
        ),
        "appeal_to_emotion": (
            "This premise is logically flawed because it relies on emotional "
            "manipulation rather than logical reasoning or evidence to support "
            "the claim."
        ),
        "circular_reasoning": (
            "This premise is logically flawed because it assumes the truth of "
            "the conclusion within the premise itself, providing no independent "
            "evidence."
        ),
        "equivocation": (
            "This premise is logically flawed because it uses ambiguous language "
            "or shifts the meaning of a key term mid-argument to draw a misleading "
            "conclusion."
        ),
        "fallacy_of_credibility": (
            "This premise is logically flawed because it misuses or fabricates "
            "authority and credibility to support a claim that the evidence does "
            "not substantiate."
        ),
        "fallacy_of_extension": (
            "This premise is logically flawed because it distorts or exaggerates "
            "the opposing position beyond what was actually claimed, making it "
            "easier to attack."
        ),
        "fallacy_of_logic": (
            "This premise is logically flawed because it draws a conclusion that "
            "does not follow from the stated premises, violating basic principles "
            "of valid reasoning."
        ),
        "fallacy_of_relevance": (
            "This premise is logically flawed because it introduces information "
            "that is irrelevant to the conclusion, distracting from the actual "
            "issue at hand."
        ),
        "false_causality": (
            "This premise is logically flawed because it assumes a causal "
            "relationship between events without sufficient evidence to establish "
            "that one actually caused the other."
        ),
        "false_dilemma": (
            "This premise is logically flawed because it presents only two options "
            "as if they are the only possibilities, ignoring other viable "
            "alternatives."
        ),
        "intentional": (
            "This premise is logically flawed because it deliberately uses "
            "misleading rhetoric or deceptive framing to manipulate the conclusion."
        ),
        "faulty_generalization": (
            "This premise is logically flawed because it draws a broad conclusion "
            "from insufficient, unrepresentative, or cherry-picked evidence."
        ),
    }

    def format_for_llm(self, result):
        """
        Format detection result to match the SFT training data schema.

        Training data format: "Fallacy Detected: {label}. {explanation why flawed}."
        No confidence scores — the model never saw them during training.
        """
        exact_fallacy_name = result["type"]

        if not result["fallacy_detected"]:
            return "Fallacy Detected: no_fallacy."

        explanation = self._FALLACY_EXPLANATIONS.get(
            exact_fallacy_name,
            "This premise is logically flawed and should be dismantled using "
            "evidence from the document.",
        )

        return f"Fallacy Detected: {exact_fallacy_name}. {explanation}"


if __name__ == "__main__":
    detector = FallacyDetector()
    test_args = [
        "You can't trust climate scientists because they just want grant money.",
        "The earth is round."
    ]

    for arg in test_args:
        print(f"\nArgument: {arg}")
        result = detector.detect(arg)
        print(f"  Detected: {result['fallacy_detected']}")
        print(f"  Type: {result['type']} ({result['confidence']:.1%})")
        print(f"  LLM context: {detector.format_for_llm(result)}")