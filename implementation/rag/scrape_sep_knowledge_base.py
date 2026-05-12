"""
Comprehensive Knowledge Base Scraper for Devil's Advocate RAG

Scrapes 300+ philosophy articles from Stanford Encyclopedia of Philosophy (SEP)
covering ethics, logic, epistemology, political philosophy, metaphysics, philosophy
of science, history of philosophy, and contemporary debate topics.

Saves raw articles to: data/sep_articles/ (shared with PoC and implementation)

Topics covered:
  - Logic & Argumentation (30+ articles)
  - Epistemology (25+ articles)
  - Ethics (40+ articles)
  - Political Philosophy (30+ articles)
  - Metaphysics (25+ articles)
  - Philosophy of Mind (25+ articles)
  - Philosophy of Science (20+ articles)
  - History of Philosophy (40+ articles)
  - Aesthetics & Culture (15+ articles)
  - Philosophy of Language (15+ articles)
  - Applied & Contemporary (20+ articles)
"""

import os
import re
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Tuple
from sep_articles import SEP_ARTICLES

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTICLES_DIR = DATA_DIR / "sep_articles"
REQUEST_DELAY = 1.5  # seconds between requests (respectful scraping)
REQUEST_TIMEOUT = 30  # seconds per request

# SEP base URL
SEP_BASE_URL = "https://plato.stanford.edu/entries"


def fetch_sep_article(slug: str) -> Tuple[str, str]:
    """
    Fetch and extract plain text from a Stanford Encyclopedia of Philosophy article.
    
    Args:
        slug: Article identifier (e.g., "kant-moral")
    
    Returns:
        Tuple of (title, text). Returns ("", "") on failure.
    """
    url = f"{SEP_BASE_URL}/{slug}/"
    
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Fetch failed: {e}")
        return "", ""

    try:
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"Parse failed: {e}")
        return "", ""

    # Extract title from h1
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else slug.replace("-", " ").title()

    # SEP articles: find main content div (try multiple selectors)
    content_div = (
        soup.find("div", id="aueditable")
        or soup.find("div", id="main-text")
        or soup.find("article")
        or soup.find("div", class_="entry-content")
        or soup.find("body")
    )

    if not content_div:
        return title, ""

    # Remove non-content elements
    for tag in content_div.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Extract text
    text = content_div.get_text(separator="\n", strip=True)

    # Clean: normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return title, text


def save_article(slug: str, title: str, text: str) -> bool:
    """Save article to disk."""
    out_path = ARTICLES_DIR / f"{slug}.txt"
    
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n{text}")
        return True
    except Exception as e:
        print(f"    ✗ Save failed: {e}")
        return False


def main():
    """Download all SEP articles organized by topic."""
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- Cleanup unused files ---
    allowed_slugs = set()
    for articles in SEP_ARTICLES.values():
        allowed_slugs.update(articles)
        
    removed = 0
    for txt_file in ARTICLES_DIR.glob("*.txt"):
        if txt_file.stem not in allowed_slugs:
            txt_file.unlink()
            removed += 1
            
    if removed > 0:
        print(f"Cleaned up {removed} deprecated articles from {ARTICLES_DIR.name}/")
    # ----------------------------
    
    total_articles = sum(len(articles) for articles in SEP_ARTICLES.values())
    print(f"Target: {total_articles} articles across {len(SEP_ARTICLES)} topics.")
    print("=" * 70)

    downloaded = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    for topic, articles in SEP_ARTICLES.items():
        print(f"\n[{topic}] {len(articles)} articles")
        
        for i, slug in enumerate(articles, 1):
            out_path = ARTICLES_DIR / f"{slug}.txt"

            # Skip if already exists with substantial content
            if out_path.exists() and out_path.stat().st_size > 500:
                skipped += 1
                continue

            print(f"  {i:2d}. {slug}...", end=" ", flush=True)
            title, text = fetch_sep_article(slug)

            if not text or len(text) < 200:
                print("✗ (no content)")
                failed += 1
                continue

            if save_article(slug, title, text):
                size_kb = len(text) / 1024
                print(f"✓ ({size_kb:.1f} KB)")
                downloaded += 1
            else:
                failed += 1

            time.sleep(REQUEST_DELAY)

    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 70}")
    print(f"Download complete:")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {downloaded + skipped + failed} / {total_articles}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Location: {ARTICLES_DIR}")
    print(f"{'=' * 70}")
    
    return downloaded + skipped


if __name__ == "__main__":
    main()
