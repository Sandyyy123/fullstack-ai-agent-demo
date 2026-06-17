"""Typed request/response contracts shared by the API and (conceptually) the frontend.

Pydantic models are the single source of truth for the API surface. The same
shapes are mirrored in the frontend's fetch calls, so the contract is explicit
on both sides rather than implied.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="End-user message.")
    use_agent: bool = Field(
        default=True,
        description="If true, run the tool-calling agent loop. If false, plain RAG answer.",
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Documents to retrieve.")


class RetrievedDoc(BaseModel):
    doc_id: str
    score: float
    text: str


class ToolCall(BaseModel):
    tool: str
    arguments: dict
    result: str


class ChatResponse(BaseModel):
    answer: str
    mode: Literal["openai", "anthropic", "demo"]
    retrieved: List[RetrievedDoc] = []
    tool_calls: List[ToolCall] = []
    tokens_estimated: int = 0


class IngestRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    llm_mode: Literal["openai", "anthropic", "demo"]
    documents_indexed: int
