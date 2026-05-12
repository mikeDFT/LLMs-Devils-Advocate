"""
Build the RAG knowledge base from SEP articles. - Needs admin perms to run (admin terminal) to use chonkie for semantic chunking

Two-tier chunking:
  1. Split on markdown headers to preserve document hierarchy
  2. Semantic chunking within each section using chonkie

Indexes chunks into LanceDB with nomic-embed-text-v2-moe embeddings.
Creates both vector (cosine) and full-text search indices for hybrid retrieval.

Output: implementation/debate_rag_db/
"""

import re
import sys
from pathlib import Path

import lancedb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SEP_ARTICLES_DIR, WIKI_ARTICLES_DIR, WIKI_LISTS_DIR, RAG_DB_DIR, EMBEDDING_MODEL,
    RAG_TABLE_NAME, CHUNK_SIZE, CHUNK_SIMILARITY_THRESHOLD,
)

# Header patterns found in SEP article text files
HEADER_PATTERN = re.compile(r"^(\d+\.[\d.]*)\s+(.+)$", re.MULTILINE)
# Header patterns found in Wikipedia article text files (explaintext=1)
WIKI_HEADER_PATTERN = re.compile(r"^==+\s+(.+?)\s+==+$", re.MULTILINE)


def split_by_headers(text, article_name, source_type="SEP"):
    """Split article into sections at headers, preserving hierarchy."""
    if source_type == "Wikipedia":
        matches = list(WIKI_HEADER_PATTERN.finditer(text))
    else:
        matches = list(HEADER_PATTERN.finditer(text))

    if not matches:
        return [{"text": text.strip(), "header": article_name, "source": article_name}]

    sections = []

    # Text before first header
    if matches[0].start() > 0:
        preamble = text[:matches[0].start()].strip()
        if len(preamble) > 50:
            sections.append({
                "text": preamble,
                "header": article_name,
                "source": article_name,
            })

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        if len(section_text) < 30:
            continue

        if source_type == "Wikipedia":
            header_chain = f"{article_name} > {match.group(1).strip()}"
        else:
            header_chain = f"{article_name} > {match.group(2).strip()}"
            
        sections.append({
            "text": section_text,
            "header": header_chain,
            "source": article_name,
        })

    return sections


def semantic_chunk_section(section, chunker):
    """Apply semantic chunking to a single section. Falls back to character splitting."""
    text = section["text"]

    if chunker is not None:
        try:
            chunks = chunker.chunk(text)
            return [
                {
                    "text": chunk.text,
                    "header": section["header"],
                    "source": section["source"],
                }
                for chunk in chunks
                if len(chunk.text.strip()) > 30
            ]
        except Exception:
            pass

    # Fallback: split by sentences into ~CHUNK_SIZE char groups
    return _character_chunk(section)


def _character_chunk(section, max_chars=2048, overlap_chars=200):
    """Simple sentence-boundary chunking as a fallback."""
    text = section["text"]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append({
                "text": " ".join(current),
                "header": section["header"],
                "source": section["source"],
            })
            # Keep last sentence as overlap
            overlap = current[-1] if current else ""
            current = [overlap, sentence]
            current_len = len(overlap) + len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence)

    if current:
        chunks.append({
            "text": " ".join(current),
            "header": section["header"],
            "source": section["source"],
        })

    return chunks


def load_chunker():
    """Try to load chonkie's SemanticChunker. Returns None if unavailable."""
    try:
        from chonkie import SemanticChunker
        chunker = SemanticChunker(
            model=EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            similarity_threshold=CHUNK_SIMILARITY_THRESHOLD,
        )
        print("Using chonkie SemanticChunker")
        return chunker
    except ImportError:
        print("chonkie not installed, falling back to character-based chunking")
        return None
    except Exception as e:
        print(f"SemanticChunker init failed ({e}), falling back to character chunking")
        return None


def main():
    SOURCE_DIRS = [
        (SEP_ARTICLES_DIR, "SEP"),
        (WIKI_ARTICLES_DIR, "Wikipedia"),
        (WIKI_LISTS_DIR, "Wikipedia"),
    ]

    chunker = load_chunker()
    all_chunks = []

    for source_dir, source_type in SOURCE_DIRS:
        if not source_dir.exists():
            print(f"Skipping {source_type} directory (not found): {source_dir}")
            continue

        article_files = sorted(source_dir.glob("*.txt"))
        if not article_files:
            print(f"No .txt files found in {source_dir}")
            continue

        print(f"Found {len(article_files)} {source_type} articles in {source_dir.name}")

        for article_path in tqdm(article_files, desc=f"Chunking {source_type} articles"):
            article_name = article_path.stem.replace("_", " ").replace("-", " ").title()
            text = article_path.read_text(encoding="utf-8", errors="replace")

            sections = split_by_headers(text, article_name, source_type=source_type)
            for section in sections:
                chunks = semantic_chunk_section(section, chunker)
                all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No chunks were generated. Please check the source directories.")

    print(f"Total chunks: {len(all_chunks)}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

    # Compute all embeddings upfront (efficient batch encoding)
    texts = [c["text"] for c in all_chunks]
    prefixed = [f"search_document: {t}" for t in texts]

    print("Computing embeddings...")
    embeddings = embedder.encode(prefixed, show_progress_bar=True, batch_size=64).tolist()

    # Build records for LanceDB (list-of-dict format)
    records = [
        {
            "id": f"chunk_{i}",
            "text": chunk["text"],
            "source": chunk["source"],
            "header": chunk["header"],
            "vector": embedding,
        }
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings))
    ]

    RAG_DB_DIR.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(RAG_DB_DIR))

    # Create table (overwrite if exists for clean rebuild)
    table = db.create_table(RAG_TABLE_NAME, data=records, mode="overwrite")
    print(f"Created table '{RAG_TABLE_NAME}' with {table.count_rows()} rows")

    # Vector index (cosine) for semantic search
    print("Creating vector index (cosine)...")
    table.create_index(metric="cosine", vector_column_name="vector", replace=True)

    # Full-text search index for hybrid retrieval
    print("Creating full-text search index...")
    table.create_fts_index("text", replace=True)

    # Scalar index on source for filtered queries (e.g. search within a topic)
    print("Creating scalar index on source column...")
    table.create_scalar_index("source", replace=True)

    print(f"\nIndexed {len(all_chunks)} chunks into {RAG_DB_DIR}")
    print(f"  Table: {RAG_TABLE_NAME}")
    print(f"  Vector index: cosine distance")
    print(f"  FTS index: enabled for hybrid search")
    print(f"  Scalar index: source column")


if __name__ == "__main__":
    main()
