import chromadb 
from sentence_transformers import SentenceTransformer
from typing import List, Dict 

_EMBED_MODEL = None

def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL

CLIENT = chromadb.Client()  # in-memory, safe for free tier
COLLECTION = CLIENT.get_or_create_collection(
    name = "debate_papers", 
)

def embed_and_store(chunks: List[Dict]) ->None:

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    metadatas = [c["metadata"] for c in chunks]

    embeddings = get_embed_model().encode(
        texts, 
        normalize_embeddings = True, 
        show_progress_bar = True
    ).tolist()

    COLLECTION.upsert(
        ids = ids, 
        documents = texts, 
        embeddings = embeddings, 
        metadatas = metadatas, 
    )

    print(f"Stored {len(chunks)} chunks in chromaDB")

def reset_collection() -> None:        # ← add here, after embed_and_store
    CLIENT.delete_collection("debate_papers")
    global COLLECTION
    COLLECTION = CLIENT.get_or_create_collection(name="debate_papers")
    print("Collection reset.")

def query_vectorstore (query : str, n_results :int = 5)  -> List[Dict]:
    query_embedding = get_embed_model().encode(
        [query], 
        normalize_embeddings = True
    ).tolist()

    results = COLLECTION.query(
        query_embeddings = query_embedding, 
        n_results = n_results, 
        include = ["documents", "metadatas", "distances"]
    )

    output = []

    for i in range(len(results["documents"][0])):
        output.append({
            "chunk_id": results["ids"][0][i],

            "text": results["documents"][0][i],

            "metadata": results["metadatas"][0][i],

            "score": 1 - results["distances"][0][i]
        })
    return output

if __name__ == "__main__":
    from retriever import fetch_papers
    from chunker import chunk_papers

    papers = fetch_papers("RAG vs fine-tuning large language models")
    chunks = chunk_papers(papers)
    embed_and_store(chunks)

    results = query_vectorstore("advantages of RAG over fine-tuning")

    for r in results:
        print(r["score"], "|", r["metadata"]["title"])
        
