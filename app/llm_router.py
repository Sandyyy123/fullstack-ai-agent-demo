"""LLM provider router with a deterministic demo fallback.

Order of preference: OpenAI -> Anthropic -> demo. The demo responder is
deterministic and dependency-free so the repo runs end to end with no API keys
(useful for CI and for a reviewer cloning the repo). In production you set one
key and the same `generate()` call uses the live model.
"""
from __future__ import annotations

import os
from typing import List, Tuple


def detect_mode() -> str:
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "demo"


def _demo_generate(prompt: str, context: List[str]) -> str:
    """Grounded, deterministic answer built from retrieved context.

    Stands in for a real LLM call. It quotes the top retrieved passage so the
    RAG path is observably working without an API key.
    """
    if context:
        lead = context[0].strip()
        return (
            "[demo mode - no API key set] Based on the retrieved context: "
            f"{lead} "
            "(Set OPENAI_API_KEY or ANTHROPIC_API_KEY to answer with a live model.)"
        )
    return (
        "[demo mode - no API key set] I have no indexed context for that yet. "
        "Ingest documents via POST /ingest, then ask again."
    )


def _openai_generate(prompt: str, context: List[str]) -> str:
    from openai import OpenAI  # imported lazily so demo mode needs no dep

    client = OpenAI()
    ctx = "\n\n".join(f"- {c}" for c in context) or "(no context retrieved)"
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "Answer using ONLY the provided context. If the context "
                "is insufficient, say so. Be concise.",
            },
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {prompt}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def _anthropic_generate(prompt: str, context: List[str]) -> str:
    import anthropic  # imported lazily

    client = anthropic.Anthropic()
    ctx = "\n\n".join(f"- {c}" for c in context) or "(no context retrieved)"
    resp = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=512,
        system="Answer using ONLY the provided context. If the context is "
        "insufficient, say so. Be concise.",
        messages=[{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {prompt}"}],
    )
    parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
    return "".join(parts)


def generate(prompt: str, context: List[str]) -> Tuple[str, str]:
    """Return (answer, mode). Falls back to demo mode if a live call fails."""
    mode = detect_mode()
    try:
        if mode == "openai":
            return _openai_generate(prompt, context), "openai"
        if mode == "anthropic":
            return _anthropic_generate(prompt, context), "anthropic"
    except Exception as exc:  # never crash the request on a provider error
        return (
            f"[fell back to demo mode after a provider error: {exc}] "
            + _demo_generate(prompt, context),
            "demo",
        )
    return _demo_generate(prompt, context), "demo"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for cost tracing in the demo."""
    return max(1, len(text) // 4)
