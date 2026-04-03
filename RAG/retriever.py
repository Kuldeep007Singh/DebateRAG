import arxiv
import requests
from dataclasses import dataclass
from typing import List

@dataclass 

class Paper:
    title : str
    abstract : str
    authors : List[str]
    url : str
    paper_id : str

def fetch_papers (topic : str, max_results = 15) -> List[Paper]:
    client = arxiv.Client(
    num_retries  = 5,
    delay_seconds= 3.0
    )

    filtered_query = f"cat:cs.CL OR cat:cs.AI {topic}"
    
    search = arxiv.Search(
        query = topic, 
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance,
        id_list     = []
    )

    papers = []

    for result in client.results(search):
        paper = Paper(
            title = result.title, 
            abstract = result.summary,
            authors = [a.name for a in result.authors], 
            url = result.entry_id,
            paper_id = result.get_short_id()
        )
        papers.append(paper)
    return papers

if __name__ == "__main__":
    results = fetch_papers("cat:cs.CL OR cat:cs.AI retrieval augmented generation fine-tuning", max_results = 5)
    for p in results:
        print(p.title, "|" , p.url)

