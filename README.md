# Agentic AI

Hands-on LangGraph projects exploring core agentic AI patterns — sequential pipelines, conditional routing, parallel branching with reducers, tool-using loops, and human-in-the-loop workflows.

## What's inside

### `states.py`
Not a runnable graph — a reference file exploring the three ways LangGraph state can be defined: `TypedDict`, `Pydantic BaseModel` (with a field validator), and Python `dataclass`.

### `sequential_base.py`
A linear 3-stage pipeline: raw text is cleaned up by an **editor** node, turned into an engaging video-script hook by a **scriptwriter** node, and localized into Hinglish by a **translator** node — each stage feeding the next via `StateGraph` edges.

### `parallel_reducers.py`
Runs three independent safety-analysis branches **in parallel** from `START` — toxicity, copyright/originality risk, and cultural sensitivity — each scoring a piece of text 0–100. Demonstrates a custom **reducer function** (`merge_score_dicts`) so all three branches can safely write into the same `safety_scores` state key without overwriting each other.

### `conditional_RAG.py`
A terminal chatbot for a college assistant. A **classifier node** categorizes each student query as `academic`, `fee`, or `general`, then `add_conditional_edges` routes it to the matching retriever — one FAISS-backed RAG retriever built from an academics handbook PDF, another from a fee-structure PDF — or straight to a general-knowledge response node if no document lookup is needed.

### `app.py`
The same conditional-RAG college assistant as above, rebuilt as a **Streamlit** chat app with a programme selector (BCA/BBA/B.Com), styled query-type badges, chat history, and a cached graph/resource build.

### `humanintheloop.py`
A LinkedIn post generator with a **human-in-the-loop** review cycle. A writer node drafts a post, then `interrupt()` pauses the graph and hands control to a human reviewer in the terminal — who can approve it or type feedback to trigger a rewrite. Uses `MemorySaver` as a checkpointer and `Command(resume=...)` to continue the graph after each human response, looping up to 3 attempts.

### `iterative_tools.py`
The same LinkedIn post generator, but self-correcting instead of human-reviewed. The writer node can call a **Tavily web search tool** (via `ToolNode` and conditional routing) to gather current info before drafting, and a separate **reviewer LLM** grades the draft against a strict rubric (hook, takeaway, length, tone, no hashtags) — looping the writer back with feedback until approved or capped at 3 attempts.

## Tech Stack

- **Framework:** LangGraph, LangChain
- **LLM Provider:** Groq
- **Vector Store:** FAISS
- **Embeddings:** HuggingFace / Sentence-Transformers
- **Search Tool:** Tavily
- **UI:** Streamlit
- **Other:** PyPDF, python-dotenv, Pydantic

## Setup

```bash
git clone <repo-url>
cd Agentic-AI
pip install -r Requirements.txt
```

Create a `.env` file with:

```
GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

`conditional_RAG.py` and `app.py` expect `academics_handbook.pdf` and `fee_structure.pdf` in the working directory — supply your own PDFs with those names to run them.