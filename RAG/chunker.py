# RAG/chunker.py

import nltk
from typing import List, Dict
from RAG.retriever import Paper

# Download sentence tokenizer on first run
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

def chunk_text(
    text       : str,
    max_words  : int = 60,   # smaller chunks = more precise retrieval
    overlap    : int = 1      # overlap in sentences, not words
) -> List[str]:
    """
    Sentence-aware chunking — never splits mid-sentence.
    Groups sentences into chunks of max_words, with sentence-level overlap.
    """
    sentences = nltk.sent_tokenize(text)
    chunks    = []
    current   = []
    count     = 0

    for i, sentence in enumerate(sentences):
        words  = sentence.split()
        count += len(words)
        current.append(sentence)

        if count >= max_words:
            chunks.append(" ".join(current))
            # overlap: carry last N sentences into next chunk
            current = current[-overlap:] if overlap else []
            count   = sum(len(s.split()) for s in current)

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_papers(papers: List[Paper]) -> List[Dict]:
    all_chunks = []

    for paper in papers:
        text_to_chunk = paper.title + ". " + paper.abstract
        chunks        = chunk_text(text_to_chunk)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{paper.paper_id}-chunk-{i}",
                "text": chunk,

                "metadata": {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "url": paper.url,
                    "authors": paper.authors,
                    "source": paper.source
                }
            })

    return all_chunks


if __name__ == "__main__":
    from RAG.retriever import fetch_papers
    papers = fetch_papers("RAG vs fine-tuning large language models")
    chunks = chunk_papers(papers)
    print(f"Total chunks: {len(chunks)}")
    print(f"Sample chunk:\n{chunks[0]['text']}")