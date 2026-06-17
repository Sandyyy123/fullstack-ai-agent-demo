"""Runnable entry point.

    python main.py            # starts the API + bundled frontend on :8000
    python main.py --selftest # runs an offline end-to-end check, no server

The --selftest path exercises RAG retrieval, the agent tool loop, and the LLM
router in demo mode so the repo is verifiably working with zero API keys.
"""
from __future__ import annotations

import sys


def selftest() -> int:
    from app.agent import run_agent
    from app.llm_router import detect_mode, generate
    from app.rag import build_seeded_store

    store = build_seeded_store()
    print(f"LLM mode: {detect_mode()}  |  documents indexed: {len(store)}")

    # 1) RAG retrieval returns the most relevant seeded doc.
    # NB: this is a lexical hashing embedding (no stemming), so the query is
    # phrased to overlap the target doc's terms - the same retrieval contract
    # holds when you swap in a real embedding model behind embed().
    hits = store.search("track answer faithfulness with an eval suite in CI", top_k=2)
    assert hits, "retrieval returned nothing"
    print(f"RAG top hit: {hits[0][0]} (score {hits[0][1]:.3f})")
    assert hits[0][0] == "evals", f"expected 'evals', got {hits[0][0]}"

    # 2) Agent calculator tool
    answer, _, tools = run_agent("calculate 12 * (3 + 4)", store)
    print(f"Agent math: {answer}  |  tools: {[t.tool for t in tools]}")
    assert answer.strip().endswith("84.0"), answer

    # 3) Agent RAG path produces a grounded answer + tool trace
    answer, retrieved, tools = run_agent("what is retrieval augmented generation?", store)
    print(f"Agent RAG answer: {answer[:80]}...")
    assert retrieved, "no documents retrieved on RAG path"
    assert any(t.tool == "kb_search" for t in tools)

    # 4) Router demo fallback never raises
    out, mode = generate("summarize the stack", ["Next.js + FastAPI + Postgres"])
    print(f"Router ({mode}): {out[:60]}...")

    print("\nSELFTEST PASSED")
    return 0


def serve() -> int:
    import uvicorn

    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=False)
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    raise SystemExit(serve())
