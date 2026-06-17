"""Minimal but real RAG retriever.

Uses a hashing bag-of-words embedding + cosine similarity so the demo runs with
zero external services or API keys. The interface (embed -> upsert -> search) is
the same one you would back with pgvector / Pinecone / Chroma in production; swap
`InMemoryVectorStore` for a real client and the rest of the app is unchanged.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
EMBED_DIM = 512


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic hashing embedding with sublinear term weighting.

    Not a transformer embedding - but a genuine vector with cosine geometry,
    enough to demonstrate retrieval end to end and to be swapped for a real
    embedding model behind the same function signature.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        idx = (hash(tok) % dim + dim) % dim
        vec[idx] += 1.0
    # sublinear scaling dampens repeated terms
    vec = np.sign(vec) * np.log1p(np.abs(vec))
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@dataclass
class _Record:
    doc_id: str
    text: str
    vector: np.ndarray


@dataclass
class InMemoryVectorStore:
    """A tiny vector store with cosine search. Stand-in for pgvector/Pinecone."""

    _records: List[_Record] = field(default_factory=list)

    def upsert(self, doc_id: str, text: str) -> None:
        vector = embed(text)
        for i, rec in enumerate(self._records):
            if rec.doc_id == doc_id:
                self._records[i] = _Record(doc_id, text, vector)
                return
        self._records.append(_Record(doc_id, text, vector))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        if not self._records:
            return []
        q = embed(query)
        scored = []
        for rec in self._records:
            score = float(np.dot(q, rec.vector))  # both unit-norm -> cosine
            scored.append((rec.doc_id, score, rec.text))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._records)


SEED_DOCS = {
    "rag-overview": (
        "Retrieval augmented generation grounds an LLM in retrieved documents. "
        "The retriever finds the most relevant chunks by vector similarity and "
        "the generator answers using only that context, reducing hallucination."
    ),
    "evals": (
        "AI features regress silently when prompts or models change. Track "
        "retrieval hit rate and answer faithfulness with an eval suite that runs "
        "in CI so a bad change is caught before it reaches production."
    ),
    "stack": (
        "The reference stack is a Next.js and TypeScript frontend, a FastAPI "
        "backend, PostgreSQL with a vector extension for retrieval, Redis for "
        "caching, and Docker images deployed to AWS through GitHub Actions."
    ),
    "agents": (
        "An agent loop lets the model call tools: it proposes a tool and "
        "arguments, the backend executes the tool, and the result is fed back "
        "until the model produces a final answer. Tools must be sandboxed."
    ),
}


def build_seeded_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    for doc_id, text in SEED_DOCS.items():
        store.upsert(doc_id, text)
    return store
