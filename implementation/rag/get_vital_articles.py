"""
Fetches the titles of Wikipedia Vital Articles from a specified level
and generates wikipedia_vital_articles.py for the scraper to use.
"""

import sys
import requests
from pathlib import Path

VITAL_PAGES = ["Wikipedia:Vital articles/Level 3"]

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "DevilsAdvocate/1.0"

def get_links_from_page(page_title):
    """Fetch all article links from a given Wikipedia page, handling pagination."""
    titles = []
    params = {
        "action": "query",
        "format": "json",
        "prop": "links",
        "titles": page_title,
        "plnamespace": 0, # Only main namespace (actual articles)
        "pllimit": "max"
    }
    
    headers = {"User-Agent": USER_AGENT}
    
    while True:
        resp = requests.get(API_URL, params=params, headers=headers)
        data = resp.json()
        
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_info in pages.items():
            if "links" in page_info:
                for link in page_info["links"]:
                    titles.append(link["title"])
                    
        # Check for pagination
        if "continue" in data:
            params.update(data["continue"])
        else:
            break
            
    return titles

def main():
    print(f"Fetching links from: {', '.join(VITAL_PAGES)}")
    all_titles = set()
    
    for page in VITAL_PAGES:
        print(f"  Fetching {page}...")
        links = get_links_from_page(page)
        all_titles.update(links)
        print(f"    Found {len(links)} links.")
        
    all_titles = sorted(list(all_titles))
    print(f"Total unique vital articles found: {len(all_titles)}")
    
    # Generate the python file
    out_path = Path(__file__).parent / "wikipedia_vital_articles.py"
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('"""\n')
        f.write('Auto-generated list of Wikipedia Vital Articles.\n')
        f.write('Do not edit manually. Run get_vital_articles.py to update.\n')
        f.write('"""\n\n')
        f.write('VITAL_ARTICLES = {\n')
        f.write('    "Vital Articles": [\n')
        for title in all_titles:
            # Escape quotes
            safe_title = title.replace('"', '\\"')
            f.write(f'        "{safe_title}",\n')
        f.write('    ]\n')
        f.write('}\n')
        
    print(f"Saved {len(all_titles)} titles to {out_path.name}")

if __name__ == "__main__":
    main()
