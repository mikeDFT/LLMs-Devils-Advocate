"""
Scrapes Wikipedia's "List of..." articles (e.g., common misconceptions)
and saves them as plain text files for the RAG knowledge base.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WIKI_LISTS_DIR
from rag.scrape_wikipedia import scrape_and_save_articles

WIKI_LISTS = [
    "List_of_common_misconceptions",
    "List_of_superseded_scientific_theories",
    "List_of_fallacies",
    "List_of_cognitive_biases",
    "List_of_conspiracy_theories",
    "List_of_pseudosciences",
]

def main():
    scrape_and_save_articles(WIKI_LISTS, WIKI_LISTS_DIR)

if __name__ == "__main__":
    main()
