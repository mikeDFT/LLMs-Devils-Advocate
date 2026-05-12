import sys
import torch
from pathlib import Path

from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RERANKER_MODEL, RERANKER_TOP_K, RERANKER_SCORE_THRESHOLD

class Reranker:
    def __init__(self, model_name=None):
        self.model_name = model_name or RERANKER_MODEL
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name,
                model_kwargs={"dtype": "auto"},
            )
        return self._model

    def rerank(self, query: str, chunks: list[dict], top_k: int = None, threshold: float = None) -> list[dict]:
        """Rerank retrieved chunks by relevance to query using cross-encoder."""
        top_k = top_k or RERANKER_TOP_K
        threshold = threshold or RERANKER_SCORE_THRESHOLD
        
        if not chunks:
            return []
        
        pairs = [[query, chunk["text"]] for chunk in chunks]
        
        # Apply sigmoid to convert logits to probabilities [0, 1]
        scores = self.model.predict(pairs, activation_fn=torch.nn.Sigmoid())
        
        # If predict returns a tensor, convert to list. If it returns a numpy array, it's fine.
        if isinstance(scores, torch.Tensor):
            scores = scores.tolist()
        
        # Attach scores to chunks
        scored = list(zip(scores, chunks))
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Filter by threshold, then take top_k
        filtered = [(s, c) for s, c in scored if s >= threshold]
        
        if not filtered:
            # If nothing passes threshold, return top-1 anyway
            return [scored[0][1]] if scored else []
        
        return [c for _, c in filtered[:top_k]]
