"""
LangGraph workflow: classify → retrieve → synthesize → cite.

The router is the headline piece of the project. Keep its prompt boring and
its decision visible — every trace should make it obvious which path ran.
"""

from __future__ import annotations

import logging
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from sec_filing_rag.extraction import get_chat_model
from sec_filing_rag.retrieval import (
    RetrievalResult,
    graph_retrieve,
    hybrid_retrieve,
    vector_retrieve,
)

log = logging.getLogger(__name__)

QueryType = Literal["semantic", "relational", "hybrid"]


class AgentState(TypedDict, total=False):
    question: str
    query_type: QueryType
    retrieval: RetrievalResult
    answer: str


# ─── Router ────────────────────────────────────────────────────────────────

_ROUTER_PROMPT = """\
Classify the user's question into ONE of three retrieval modes.

- 'semantic': summary / description / explanation of disclosed content for
  one or a few known companies. Pure passage retrieval suffices.
  Example: "Summarize Apple's AI risk disclosures."

- 'relational': requires traversing relationships between entities — auditors,
  customers, subsidiaries, competitors. Can be answered from graph metadata
  alone, without reading filing text.
  Example: "What other companies does Tesla's auditor audit?"

- 'hybrid': requires BOTH a relational step (to identify a set of entities)
  AND retrieving filing text from those entities.
  Example: "How do companies audited by Deloitte describe climate risk?"

Reply with the single word: semantic, relational, or hybrid.
"""


def classify(state: AgentState) -> AgentState:
    model = get_chat_model()
    resp = model.invoke([
        SystemMessage(_ROUTER_PROMPT),
        HumanMessage(state["question"]),
    ])
    label = str(resp.content).strip().lower().split()[0]
    if label not in {"semantic", "relational", "hybrid"}:
        log.warning("Router returned %r, defaulting to 'semantic'", label)
        label = "semantic"
    return {"query_type": label}                                  # type: ignore[return-value]


# ─── Retrieve ──────────────────────────────────────────────────────────────


def retrieve(state: AgentState) -> AgentState:
    q, qt = state["question"], state["query_type"]
    match qt:
        case "semantic":
            return {"retrieval": vector_retrieve(q)}
        case "relational":
            return {"retrieval": graph_retrieve(q)}
        case "hybrid":
            return {"retrieval": hybrid_retrieve(q)}
        case _:
            return {"retrieval": vector_retrieve(q)}


# ─── Synthesize ────────────────────────────────────────────────────────────

_SYNTHESIS_PROMPT = """\
You are a financial research assistant. Answer the user's question using ONLY
the provided context. Cite filings inline as [TICKER FY{year} Item {item}]
whenever you use a passage. If the context is insufficient, say so plainly.

Be concise. Prefer one tight paragraph over bullet lists unless the question
explicitly asks to compare or list.
"""


def _format_context(r: RetrievalResult) -> str:
    parts: list[str] = []
    if r.cypher:
        parts.append(f"=== Cypher executed ===\n{r.cypher}")
    if r.graph_facts:
        parts.append("=== Graph rows ===")
        for row in r.graph_facts[:25]:
            parts.append(str(row))
    if r.chunks:
        parts.append("=== Filing passages ===")
        for d in r.chunks:
            m = d.metadata
            tag = f"[{m.get('ticker','?')} FY{m.get('fiscal_year','?')} Item {m.get('item_no','?')}]"
            parts.append(f"{tag} {d.page_content}")
    return "\n\n".join(parts) if parts else "(no context retrieved)"


def synthesize(state: AgentState) -> AgentState:
    model = get_chat_model()
    ctx = _format_context(state["retrieval"])
    resp = model.invoke([
        SystemMessage(_SYNTHESIS_PROMPT),
        HumanMessage(f"QUESTION:\n{state['question']}\n\nCONTEXT:\n{ctx}"),
    ])
    return {"answer": str(resp.content).strip()}


# ─── Wire it up ────────────────────────────────────────────────────────────


def build_workflow() -> Any:                                       # noqa: F821 (Any avoids langgraph type plumbing)
    g = StateGraph(AgentState)
    g.add_node("classify", classify)
    g.add_node("retrieve", retrieve)
    g.add_node("synthesize", synthesize)
    g.set_entry_point("classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


# Top-level convenience used by the CLI and UI.
def answer(question: str) -> AgentState:
    return build_workflow().invoke({"question": question})        # type: ignore[no-any-return]


# Annoying typing import done late so it doesn't shadow the public name.
from typing import Any  # noqa: E402
