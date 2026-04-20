# RAG/retriever.py

import arxiv
import requests
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Paper:
    title     : str
    abstract  : str
    authors   : List[str]
    url       : str
    paper_id  : str
    source    : str = "arxiv"   # new field — tracks where paper came from


# ── ArXiv Fetch ───────────────────────────────────────────
def fetch_from_arxiv(topic: str, max_results: int = 15) -> List[Paper]:
    try:
        client = arxiv.Client(
            num_retries   = 3,
            delay_seconds = 5.0    # increased from 3.0
        )

        search = arxiv.Search(
            query       = topic,
            max_results = max_results,
            sort_by     = arxiv.SortCriterion.Relevance,
        )

        papers = []
        for result in client.results(search):
            papers.append(Paper(
                title     = result.title,
                abstract  = result.summary,
                authors   = [a.name for a in result.authors],
                url       = result.entry_id,
                paper_id  = result.get_short_id(),
                source    = "arxiv"
            ))
        return papers

    except Exception as e:
        print(f"[arxiv] Failed: {e}. Falling back to Wikipedia only.")
        return []    # ← empty list triggers Wikipedia fallback automatically

# ── Wikipedia Fetch ───────────────────────────────────────
def fetch_from_wikipedia(topic: str, max_articles: int = 5) -> List[Paper]:
    """
    Uses Wikipedia's free public API — no key needed.
    Searches for topic, fetches summaries of top articles.
    """
    papers = []

    # Step 1 — search for relevant article titles
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action"   : "query",
        "list"     : "search",
        "srsearch" : topic,
        "srlimit"  : max_articles,
        "format"   : "json"
    }

    try:
        search_resp = requests.get(search_url, params=search_params, timeout=10)
        search_data = search_resp.json()
        results     = search_data.get("query", {}).get("search", [])
    except Exception as e:
        print(f"[Wikipedia] Search failed: {e}")
        return []

    # Step 2 — fetch summary for each article found
    for item in results:
        title    = item["title"]
        page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

        summary_params = {
            "action"      : "query",
            "titles"      : title,
            "prop"        : "extracts",
            "exintro"     : True,      # intro section only
            "explaintext" : True,      # plain text, no HTML
            "format"      : "json"
        }

        try:
            summary_resp = requests.get(search_url, params=summary_params, timeout=10)
            pages        = summary_resp.json()["query"]["pages"]
            page         = next(iter(pages.values()))
            extract      = page.get("extract", "").strip()

            if not extract:
                continue

            papers.append(Paper(
                title     = title,
                abstract  = extract[:2000],  # cap at 2000 chars
                authors   = ["Wikipedia Contributors"],
                url       = page_url,
                paper_id  = f"wiki-{title.replace(' ', '_')[:40]}",
                source    = "wikipedia"
            ))

        except Exception as e:
            print(f"[Wikipedia] Failed to fetch '{title}': {e}")
            continue

    return papers


# ── Main Entry Point ──────────────────────────────────────
def fetch_papers(
    topic       : str,
    max_results : int = 15,
    min_papers  : int = 3          # fallback triggers if arxiv returns less than this
) -> List[Paper]:
    """
    Primary fetch from arxiv.
    If arxiv returns fewer than min_papers, Wikipedia fills the gap.
    """
    print(f"[Retriever] Fetching arxiv papers for: '{topic}'")
    papers = fetch_from_arxiv(topic, max_results=max_results)

    if len(papers) >= min_papers:
        print(f"[Retriever] arxiv returned {len(papers)} papers. No fallback needed.")
        return papers

    # Fallback
    print(f"[Retriever] arxiv returned only {len(papers)} papers. Triggering Wikipedia fallback...")
    wiki_papers = fetch_from_wikipedia(topic, max_articles=5)

    combined = papers + wiki_papers
    print(f"[Retriever] Total after fallback: {len(combined)} sources")
    return combined


if __name__ == "__main__":
    # Test with a research topic — should use arxiv only
    print("=== Research Topic ===")
    results = fetch_papers("retrieval augmented generation", max_results=5)
    for p in results:
        print(f"[{p.source}] {p.title}")

    print("\n=== General Topic ===")
    results = fetch_papers("should remote work become permanent", max_results=5)
    for p in results:
        print(f"[{p.source}] {p.title}")