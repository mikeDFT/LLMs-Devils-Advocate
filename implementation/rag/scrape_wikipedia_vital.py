"""
Scrapes the Wikipedia Vital Articles defined in wikipedia_vital_articles.py
and saves them as plain text files for the RAG knowledge base.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WIKI_ARTICLES_DIR
from rag.wikipedia_vital_articles import VITAL_ARTICLES
from rag.scrape_wikipedia import scrape_and_save_articles

def main():
    all_titles = []
    for category, titles in VITAL_ARTICLES.items():
        all_titles.extend(titles)
        
    scrape_and_save_articles(all_titles, WIKI_ARTICLES_DIR)

if __name__ == "__main__":
    main()
