"""
Centralized Wikipedia scraping logic for RAG data preparation.
"""

import time
import os
from pathlib import Path
import requests

API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_SCRAPING_EMAIL = os.environ.get("WIKI_SCRAPING_EMAIL", "default@example.com")
USER_AGENT = f"DevilsAdvocate/1.0 ({WIKI_SCRAPING_EMAIL})"

def get_wikipedia_text(titles, session=None):
    """Fetches plain text for a list of Wikipedia titles using the MediaWiki API."""
    headers = {"User-Agent": USER_AGENT}

    titles_str = "|".join(titles)

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": titles_str,
        "explaintext": 1,
        "redirects": 1,
        "exlimit": "max",
        "maxlag": 5
    }
    
    req_session = session or requests.Session()
    
    while True:
        try:
            response = req_session.get(API_URL, params=params, headers=headers)
            
            if response.status_code == 429 or "Retry-After" in response.headers:
                retry_after = int(response.headers.get("Retry-After", 5))
                print(f"Server overloaded or maxlag reached. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            if "error" in data and data["error"].get("code") == "maxlag":
                retry_after = int(response.headers.get("Retry-After", 5))
                print(f"Maxlag error. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
                
            break
        except Exception as e:
            print(f"Failed to fetch batch: {e}")
            return {}
        
    articles = {}
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        title = page_data.get("title")
        extract = page_data.get("extract")
        if title and extract:
            articles[title] = extract
            
    return articles

def scrape_and_save_articles(titles, out_dir):
    """Batches Wikipedia titles, fetches their text, and saves to the output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(titles)} articles to process.")
    
    pending_titles = []
    for title in titles:
        safe_title = title.replace("/", "_").replace(" ", "_")
        out_path = out_dir / f"{safe_title}.txt"
        if not out_path.exists():
            pending_titles.append(title)
        else:
            print(f"Skipping {title} (already exists)...")
            
    if not pending_titles:
        print("All articles already downloaded.")
        return
        
    print(f"Need to fetch {len(pending_titles)} articles.")

    with requests.Session() as session:
        for i, title in enumerate(pending_titles):
            print(f"Fetching {title} ({i + 1}/{len(pending_titles)})...")
            
            articles = get_wikipedia_text([title], session=session)
            
            for article_title, text in articles.items():
                safe_title = article_title.replace("/", "_").replace(" ", "_")
                out_path = out_dir / f"{safe_title}.txt"
                
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
                    
                print(f"  Saved {article_title}")
                
            time.sleep(1)
