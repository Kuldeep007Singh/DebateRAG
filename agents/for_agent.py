from groq import Groq
from typing import List, Dict
from RAG.vectorstore import query_vectorstore
import os



from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


SYSTEM_PROMPT = """
You are a professional debate agent arguing FOR the given topic.
Your job is to build a strong, evidence-based argument supporting the topic.
Rules:
- Every claim must reference the provided context chunks
- Cite the paper title and authors for each claim
- Be concise, structured, and persuasive
- Do not make up any facts outside the provided context
"""

def build_context (chunks : List[Dict])  -> str:
    context = ""
    for i, chunk in enumerate(chunks):
        context += f""" 
[{i+1}] Title : {chunk['metadata'] ['title']}
    Authors = {chunk['metadata']['authors']}
    URL = {chunk['metadata']['url']}
    Text = {chunk['text']}
---"""
    return context

def run_for_agent(topic: str ) -> Dict: 
    chunks = query_vectorstore(
        query = f"evidence supporting {topic}",
        n_results = 5
    )
    context = build_context(chunks)

    user_message = f"""
Topic: {topic}

Here is the research evidence you must use to build your argument:

{context}

Now construct a strong FOR argument using only this evidence.
Structure your response as:
1. Opening claim
2. Evidence point 1 (with citation)
3. Evidence point 2 (with citation)
4. Evidence point 3 (with citation)
5. Closing statement
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )   
    return {
        "role"  : "FOR", 
        "argument" : response.choices[0].message.content, 
        "chunks" : chunks
    }

if __name__ == "__main__":
    result = run_for_agent("RAG is better than fine-tuning for enterprise LLMs")
    print(result["argument"])

