from typing import List, Dict
from RAG.retriever import Paper

def chunk_text( text : str, chunk_size : int =  500, overlap : int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size 
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = start + chunk_size - overlap
    
    return chunks

def chunk_papers (papers : List[Paper]) -> List[Dict]:
    all_chunks = []

    for paper in papers: 
        text_to_chunk = paper.title + " " + paper.abstract
        chunks = chunk_text(text_to_chunk)

        for i, chunk in enumerate(chunks): 
            all_chunks.append({
                "chunk_id" : f"{paper.paper_id} - chunk - {i}", 
                "text" : chunk,
                "paper_id" : paper.paper_id,
                "title" : paper.title, 
                "url" : paper.url, 
                "authors" : paper.authors
            })
        
    return all_chunks

if __name__ == "__main__":
    from retriever import fetch_papers
    papers = fetch_papers("RAG vs fine tuning large language models")
    chunks = chunk_papers(papers)
    print(f"total chunks created : {len(chunks)}")
    print(chunks[0])