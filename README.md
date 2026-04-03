# ⚖️ DebateRAG — Multi-Agent Research Debate System

A multi-agent AI system that debates any research topic using real academic papers. Two specialized agents argue FOR and AGAINST the topic, rebut each other, and an impartial judge delivers a structured verdict — all grounded in semantically retrieved research evidence.

---

## 🧠 System Overview

```
Topic Input
    ↓
Papers fetched & embedded into ChromaDB
    ↓
FOR Agent → AGAINST Agent → FOR Rebuttal → AGAINST Rebuttal → Judge
    ↓
Structured Verdict with Confidence Score
```

Every argument is grounded in retrieved paper chunks. Agents cannot generate claims outside the provided context — citations are mandatory.

---

## 🏗️ Architecture

```
DebateRAG/
├── agents/
│   ├── for_agent.py          # Argues FOR the topic using retrieved evidence
│   ├── against_agent.py      # Argues AGAINST the topic using retrieved evidence
│   ├── judge_agent.py        # Evaluates full debate, delivers scored verdict
│   └── orchestrator.py       # LangGraph state machine managing agent flow
├── RAG/
│   ├── retriever.py          # Fetches academic papers
│   ├── chunker.py            # Splits papers into semantic chunks
│   └── vectorstore.py        # BGE embeddings + ChromaDB storage and retrieval
├── api/
│   └── main.py               # FastAPI backend
├── frontend/
│   └── app.py                # Streamlit UI
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA 3.3 70B via Groq |
| Embeddings | BGE-large-en-v1.5 (BAAI) |
| Vector Store | ChromaDB |
| Agent Orchestration | LangGraph |
| Backend | FastAPI |
| Frontend | Streamlit |
| Language | Python 3.11 |

---

## ⚙️ Installation

```bash
git clone https://github.com/Kuldeep007Singh/DebateRAG.git
cd DebateRAG
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Created a `.env` file in the project root:
```
GROQ_API_KEY=GROQ_API_KEY=your_groq_api_key
```

---

## 🚀 Running

**Backend:**
```bash
python -m uvicorn api.main:app --reload
```

**Frontend:**
```bash
streamlit run frontend/app.py
```

---

## 🔄 Agent Flow

Built on LangGraph's `StateGraph` — each agent is a node that reads from and writes to a shared `DebateState` object. The graph executes deterministically in sequence:

```
for_agent → against_agent → for_rebuttal → against_rebuttal → judge → END
```

Each node performs its own RAG retrieval before generating — the FOR and AGAINST agents query for supporting and opposing evidence respectively, ensuring the debate is genuinely adversarial rather than both sides drawing from the same framing.

---

## 📊 Output

```json
{
  "topic": "...",
  "for_argument": "...",
  "against_argument": "...",
  "for_rebuttal": "...",
  "against_rebuttal": "...",
  "verdict": "..."
}
```

Full debate report is downloadable as JSON from the UI.

---

## 🔑 Design Decisions

**BGE-large-en-v1.5** was chosen for its strong performance on semantic similarity in academic text — significantly outperforms smaller embedding models on retrieval tasks.

**LangGraph over plain LangChain** gives explicit control over agent execution order and shared state. Each node's output feeds directly into the next agent's context, making the debate flow traceable and debuggable.

**Collection reset per debate run** prevents chunk accumulation across multiple runs on different topics — each debate starts with a clean ChromaDB collection scoped to the current topic's papers.

---

## 👨‍💻 Author

**Kuldeep Singh** — AI/ML Engineer  
MCA Graduate, University of Rajasthan  
[GitHub](https://github.com/Kuldeep007Singh)