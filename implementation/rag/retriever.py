"""
RAG retrieval interface for the debate knowledge base.

Supports vector, full-text, and hybrid (vector + FTS) search via LanceDB.

Usage:
    from rag.retriever import RAGRetriever
    rag = RAGRetriever()
    results = rag.query("What is Kant's categorical imperative?")
"""

import sys
from pathlib import Path

import lancedb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAG_DB_DIR, EMBEDDING_MODEL, RAG_TABLE_NAME, RAG_TOP_K


class RAGRetriever:
    def __init__(self, db_path=None, top_k=None):
        self.db_path = str(db_path or RAG_DB_DIR)
        self.top_k = top_k or RAG_TOP_K
        self._embedder = None
        self._table = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
        return self._embedder

    @property
    def table(self):
        if self._table is None:
            db = lancedb.connect(self.db_path)
            self._table = db.open_table(RAG_TABLE_NAME)
        return self._table

    def format_results(self, results):
        """Format a list of raw result dicts into a single string."""
        if not results:
            return "No relevant evidence found in the knowledge base."

        formatted = []
        for row in results:
            source = row.get("source", "Unknown")
            header = row.get("header", "")
            formatted.append(f"[{source} — {header}]\n{row['text']}")

        return "\n\n---\n\n".join(formatted)

    def query(self, query_text, n_results=None):
        """Return formatted string of top results using hybrid search (vector + FTS)."""
        results = self.query_raw(query_text, n_results)
        return self.format_results(results)

    def query_multi(self, queries: list[str], n_results_per_query: int = None):
        """Execute multiple queries, merge and deduplicate results."""
        n_results_per_query = n_results_per_query or 2
        all_results = []
        seen_texts = set()

        for q in queries:
            results = self.query_raw(q, n_results_per_query)
            for r in results:
                text = r.get("text", "")
                if text not in seen_texts:
                    seen_texts.add(text)
                    all_results.append(r)

        return all_results

    def query_raw(self, query_text, n_results=None):
        """Return raw LanceDB results as list of dicts.

        Uses hybrid search (vector + FTS) with RRF reranking.
        Falls back to vector-only search if FTS index is unavailable.
        """
        n = n_results or self.top_k
        # nomic-v2 expects "search_query: " prefix for queries
        prefixed = f"search_query: {query_text}"
        embedding = self.embedder.encode(prefixed).tolist()

        try:
            # Hybrid search: vector similarity + full-text BM25, reranked with RRF
            results = (
                self.table.search(embedding, query_type="hybrid")
                .text(query_text)
                .metric("cosine")
                .select(["text", "source", "header", "_distance"])
                .limit(n)
                .to_list()
            )
        except Exception:
            # Fallback: vector-only search if FTS index is missing
            results = (
                self.table.search(embedding)
                .metric("cosine")
                .select(["text", "source", "header", "_distance"])
                .limit(n)
                .to_list()
            )

        return results


if __name__ == "__main__":
    rag = RAGRetriever()
    test_queries = [
        "What is the categorical imperative?",
        "Arguments for and against free will",
        "Types of logical fallacies",
        "Utilitarianism vs deontological ethics",
        "What is the Chinese Room argument?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        result = rag.query(q, n_results=3)
        print(result[:500] + "..." if len(result) > 500 else result)
