# Full-Stack AI Agent Demo

A compact, **runnable** reference for an AI-powered full-stack feature: a typed
React frontend talking to a FastAPI backend, with an LLM layer that does
**RAG retrieval + tool-calling agents + evals**, and a CI self-test that runs
with **zero API keys**.

Built as a working sample of how I ship AI features end to end (concept →
production), not a slide deck.

## Architecture

```
┌─────────────────────┐     POST /chat (+ /chat/stream SSE)     ┌──────────────────────────┐
│  Frontend            │ ───────────────────────────────────▶  │  FastAPI backend          │
│  React + TS (CDN)    │ ◀───────────────────────────────────  │  app/server.py            │
│  frontend/index.html │     typed JSON (Pydantic contracts)    │                           │
└─────────────────────┘                                        │  ┌──────────────────────┐ │
                                                                │  │ Agent loop           │ │
                                                                │  │ app/agent.py         │ │
                                                                │  │  • route → tool      │ │
                                                                │  │  • calculator (safe) │ │
                                                                │  │  • kb_search         │ │
                                                                │  └──────────┬───────────┘ │
                                                                │             ▼              │
                                                                │  ┌──────────────────────┐ │
                                                                │  │ RAG retriever        │ │
                                                                │  │ app/rag.py           │ │
                                                                │  │  vector store +      │ │
                                                                │  │  cosine search       │ │
                                                                │  └──────────┬───────────┘ │
                                                                │             ▼              │
                                                                │  ┌──────────────────────┐ │
                                                                │  │ LLM router           │ │
                                                                │  │ app/llm_router.py    │ │
                                                                │  │ OpenAI→Anthropic→demo│ │
                                                                │  └──────────────────────┘ │
                                                                └──────────────────────────┘
```

**Production mapping:** swap `InMemoryVectorStore` for pgvector / Pinecone /
Chroma, point the router at your provider key, and containerize — the rest of
the app is unchanged. Layers are deliberately isolated behind small interfaces.

## Quick start

```bash
pip install -r requirements.txt

# Verify everything works end to end with NO API key (used by CI):
python main.py --selftest

# Run the API + bundled frontend:
python main.py
# open http://localhost:8000
```

Run against a live model by copying `.env.example` to `.env` and setting one of
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. With no key set, the app runs in a
deterministic **demo mode** that still exercises retrieval and the agent loop.

## API

| Method | Path           | Purpose                                            |
|--------|----------------|----------------------------------------------------|
| GET    | `/health`      | Status, active LLM mode, document count            |
| POST   | `/ingest`      | Add a document to the vector store                 |
| POST   | `/chat`        | RAG + agent answer with retrieved docs + tool trace|
| POST   | `/chat/stream` | Same, streamed as Server-Sent Events               |

Every `/chat` response returns the **retrieved documents, the tools the agent
called, and a token estimate** — so the AI layer is observable, not a black box.

## What this demonstrates

- **Full stack:** React/TS frontend ⇄ FastAPI backend with shared typed contracts
- **RAG:** embed → vector store → cosine retrieval, swappable for a real vector DB
- **Agents:** a tool-calling loop with a *sandboxed* calculator (AST, never `eval`) and a KB tool
- **Provider-agnostic LLM layer:** OpenAI / Anthropic / demo fallback behind one `generate()`
- **Evals + CI:** `--selftest` asserts retrieval correctness and the agent paths; wired into GitHub Actions
- **Observability:** retrieved docs, tool calls, and token estimates surfaced on every response

## Layout

```
main.py                 # entry point: serve | --selftest
app/server.py           # FastAPI routes + SSE streaming
app/agent.py            # tool-calling loop + sandboxed tools
app/rag.py              # embeddings + in-memory vector store
app/llm_router.py       # OpenAI/Anthropic/demo router
app/schemas.py          # Pydantic request/response contracts
frontend/index.html     # React chat UI (CDN, no build step)
.github/workflows/ci.yml# runs the self-test on every push
```
