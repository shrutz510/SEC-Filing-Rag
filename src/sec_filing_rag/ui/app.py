"""
Streamlit demo. Run with:
    streamlit run src/sec_filing_rag/ui/app.py

Shows the router's decision, the Cypher (if any), retrieved passages, and
the final synthesized answer. The visible reasoning is half the point.
"""

from __future__ import annotations

import streamlit as st

from sec_filing_rag.agent import answer

st.set_page_config(page_title="SEC Filing RAG", layout="wide")
st.title("SEC Filing RAG")
st.caption("Hybrid RAG + Knowledge Graph over SEC filings · LangChain + LangGraph")

EXAMPLES = [
    "Summarize Apple's AI-related risk disclosures in their most recent 10-K.",
    "Who audits Tesla, and what other companies do they audit?",
    "How do companies audited by Deloitte describe climate-related risk?",
    "Which suppliers are mentioned in NVIDIA's filings, and what risks do those suppliers themselves disclose?",
]

with st.sidebar:
    st.subheader("Try a question")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state["question"] = q

question = st.text_area(
    "Your question",
    value=st.session_state.get("question", ""),
    height=80,
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Routing, retrieving, synthesizing…"):
        result = answer(question.strip())

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Answer")
        st.write(result["answer"])

        if result["retrieval"].cypher:
            with st.expander("Cypher query"):
                st.code(result["retrieval"].cypher, language="cypher")

        if result["retrieval"].chunks:
            with st.expander(f"Retrieved passages ({len(result['retrieval'].chunks)})"):
                for d in result["retrieval"].chunks:
                    m = d.metadata
                    st.markdown(f"**{m.get('ticker')} FY{m.get('fiscal_year')} · Item {m.get('item_no')}**")
                    st.write(d.page_content)
                    st.divider()

        if result["retrieval"].graph_facts:
            with st.expander(f"Graph rows ({len(result['retrieval'].graph_facts)})"):
                st.json(result["retrieval"].graph_facts)

    with col_b:
        st.subheader("Route")
        st.metric("Mode", result.get("query_type", "?").upper())
