"""A small, real tool-calling agent loop.

The agent inspects the user message, decides whether a tool is needed, executes
the tool in-process, and feeds the result into the grounded answer. This mirrors
the production pattern (model proposes tool + args, backend executes, result is
returned to the model) without requiring a live model for the demo path.

Tools are intentionally sandboxed: the calculator uses a restricted AST evaluator,
never eval(); the knowledge-base tool only reads the in-memory store.
"""
from __future__ import annotations

import ast
import operator
import re
from typing import Callable, Dict, List, Tuple

from .llm_router import generate
from .rag import InMemoryVectorStore
from .schemas import RetrievedDoc, ToolCall

# ---- Tool 1: safe arithmetic ------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _safe_eval(node.operand)
        return val if isinstance(node.op, ast.UAdd) else -val
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree))
    except Exception:
        return f"could not evaluate '{expression}'"


# ---- Tool 2: knowledge base lookup -----------------------------------------


def make_kb_search(store: InMemoryVectorStore) -> Callable[[str], str]:
    def kb_search(query: str) -> str:
        hits = store.search(query, top_k=1)
        if not hits:
            return "no matching documents"
        _, score, text = hits[0]
        return f"(score {score:.3f}) {text}"

    return kb_search


# ---- Routing logic ----------------------------------------------------------

_MATH_RE = re.compile(r"^[\s\d+\-*/().%^]+$")


def _route_tool(message: str) -> Tuple[str, dict] | None:
    """Decide which tool (if any) the message needs.

    A live model would emit this via function-calling; here we use a small
    deterministic router so the loop is demonstrable without an API key.
    """
    stripped = message.strip()
    # extract a bare arithmetic expression if the user asked to compute one
    m = re.search(r"(?:calculate|compute|what is)\s+([\d+\-*/().%^\s]+)", stripped, re.I)
    if m and _MATH_RE.match(m.group(1).strip()):
        return "calculator", {"expression": m.group(1).strip()}
    if _MATH_RE.match(stripped) and any(c in stripped for c in "+-*/^%"):
        return "calculator", {"expression": stripped}
    return None


def run_agent(
    message: str,
    store: InMemoryVectorStore,
    top_k: int = 3,
) -> Tuple[str, List[RetrievedDoc], List[ToolCall]]:
    """One pass of: maybe-call-a-tool -> retrieve -> grounded answer."""
    tool_calls: List[ToolCall] = []

    routed = _route_tool(message)
    if routed:
        tool_name, args = routed
        if tool_name == "calculator":
            result = calculator(args["expression"])
            tool_calls.append(ToolCall(tool=tool_name, arguments=args, result=result))
            # arithmetic is self-contained; answer directly
            return (f"{args['expression']} = {result}", [], tool_calls)

    # RAG path
    hits = store.search(message, top_k=top_k)
    retrieved = [RetrievedDoc(doc_id=d, score=round(s, 4), text=t) for d, s, t in hits]
    context = [r.text for r in retrieved]

    # the agent can also consult the KB tool explicitly for traceability
    kb = make_kb_search(store)
    kb_result = kb(message)
    tool_calls.append(
        ToolCall(tool="kb_search", arguments={"query": message}, result=kb_result)
    )

    answer, _mode = generate(message, context)
    return answer, retrieved, tool_calls
