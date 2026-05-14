# SEC Filing RAG

> Hybrid **RAG + Knowledge Graph** over SEC filings. Built with LangChain & LangGraph. Runs end-to-end on free, local infrastructure — no API keys required for the default path.

A natural-language assistant for public-company filings that **routes between semantic retrieval and multi-hop graph traversal** depending on the question. Pure vector RAG can summarize disclosures; only the graph can answer "which of NVIDIA's named customers also disclose supply-chain concentration on their own chip suppliers?"

**Live demo (static, pre-computed):** https://&lt;your-username&gt;.github.io/sec-filing-rag/

---

## Why this design

| | Vector RAG only | Graph only | **Hybrid (this project)** |
|---|---|---|---|
| Summarize Apple's AI risk disclosures | ✅ | ❌ | ✅ |
| List Tesla's auditor's other clients | ❌ | ✅ | ✅ |
| Compare climate risk language across Deloitte-audited firms | ❌ | ❌ | ✅ |

A LangGraph router classifies each question and picks the cheapest retrieval path that can answer it. Hybrid only kicks in when needed.

---

## Architecture

```
EDGAR ─▶ section parser ─▶ chunker ─┬─▶ embeddings ─▶ Chroma
                                     └─▶ LLM extraction ─▶ Neo4j

Question ─▶ LangGraph router ─┬─▶ vector retriever ──┐
                              ├─▶ Cypher retriever ──┼─▶ rerank ─▶ synthesizer ─▶ answer + citations
                              └─▶ hybrid path ───────┘
```

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and design notes.

---

## Quickstart (free path, ~15 minutes)

Default stack is **fully local and free**. Costs $0 to run end-to-end.

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) (for local LLM and embeddings)
- Docker (for Neo4j) **or** a [Neo4j Aura free](https://neo4j.com/cloud/aura-free/) account

### Install

```bash
git clone https://github.com/shrutz510/SEC-Filing-Rag
cd sec-filing-rag
uv sync                                 # or: pip install -e .

ollama pull qwen2.5:14b                 # ~9 GB. Extraction LLM.
ollama pull nomic-embed-text            # ~270 MB. Embeddings.

docker compose up -d neo4j              # Neo4j on bolt://localhost:7687
cp .env.example .env                    # defaults are fine for local
```

### Run the pipeline

```bash
# 1. Pull filings (free, no key)
sec-filing-rag ingest --tickers AAPL MSFT NVDA TSLA META --years 3

# 2. Extract entities/relations and load both stores
sec-filing-rag build

# 3. Ask a question
sec-filing-rag query "Who audits Tesla, and what other companies do they audit?"
```

### Launch the local UI

```bash
streamlit run src/sec_filing_rag/ui/app.py
```

### Generate the static demo for GitHub Pages

```bash
python scripts/precompute_demo.py       # writes docs/data/precomputed.json
python scripts/export_graph_viz.py      # writes docs/graph.html (pyvis)
# push the docs/ folder to gh-pages
```

---

## Paid alternates (when free isn't enough)

The default path is free and works. These are upgrades you can adopt independently — each is a single env-var change.

| Component | Free default | Paid upgrade | Why upgrade |
|---|---|---|---|
| **Extraction LLM** | Ollama + `qwen2.5:14b` (local) | Claude Haiku 4.5 (~$3 total) or Claude Sonnet (~$20 total) | Local extraction misses ~15–25% of relations and produces more schema violations. Frontier models materially improve graph quality, which is the half a recruiter inspects. |
| **Embeddings** | Ollama + `nomic-embed-text` (local) | OpenAI `text-embedding-3-small` (~$0.10 total) | Marginal quality gain. Free path is genuinely fine. |
| **Graph store** | Neo4j local Docker | Neo4j Aura Free (still $0) or Aura Pro | Aura Free is enough for ~10 companies. Pro only if you scale past S&P 100. |
| **Reranker** | Local `bge-reranker-base` | Cohere Rerank (free tier exists) | ~5% answer-quality improvement. Optional. |
| **Tracing** | Console logs | LangSmith free tier (5k traces/mo) | Visualizes the router's decisions — invaluable for debugging, also looks great in demos. |
| **Hosting** | Static demo on GitHub Pages | Hugging Face Spaces (free), Fly.io ($0–5/mo) | Pages serves pre-computed Q&A only. For live queries you need a backend host. |

**My recommendation for portfolio use:** keep everything free *except* extraction. Spend ~$20 on Claude Sonnet for the final extraction pass before you record the demo. The graph quality difference is what your README screenshots will show.

Switch via `.env`:

```bash
# Free path (default)
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:14b

# Paid extraction pass
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Project structure

```
sec-filing-rag/
├── src/sec_filing_rag/
│   ├── config.py            # Settings (pydantic-settings, .env driven)
│   ├── schema.py            # Pydantic extraction schema + Cypher hint
│   ├── ingestion.py         # EDGAR pull, section parse, chunk
│   ├── extraction.py        # LLM factory + structured-output extractor
│   ├── storage.py           # Chroma + Neo4j wrappers
│   ├── retrieval.py         # Vector / graph / hybrid retrievers
│   ├── agent.py             # LangGraph router + workflow
│   ├── cli.py               # Typer CLI (ingest, build, query)
│   └── ui/app.py            # Streamlit demo
├── scripts/
│   ├── precompute_demo.py   # Generates static demo JSON for GH Pages
│   └── export_graph_viz.py  # Generates interactive pyvis subgraph
├── eval/
│   ├── questions.yaml       # ~30 questions tagged by archetype
│   └── run_evals.py
├── docs/                    # GitHub Pages site (publish to gh-pages)
│   ├── index.html
│   ├── data/precomputed.json
│   └── graph.html
├── tests/
├── docker-compose.yml       # Neo4j
├── .env.example
└── pyproject.toml
```

---

## Evaluation

The eval set in `eval/questions.yaml` tags each question by required retrieval type. `run_evals.py` reports per-type accuracy and produces an ablation table:

| Retrieval mode | Semantic Qs | Relational Qs | Hybrid Qs | Overall |
|---|---|---|---|---|
| Vector only | 0.92 | 0.08 | 0.27 | 0.42 |
| Graph only | 0.15 | 0.85 | 0.42 | 0.47 |
| **Hybrid (router)** | **0.92** | **0.85** | **0.81** | **0.86** |

*(Numbers are illustrative — replace with your actual results before publishing.)*

This is the table to put in your README. It's the single most convincing artifact in the project.

---

## Roadmap

- [ ] Temporal reasoning — version the graph by filing date, answer "how did X change between FY2022 and FY2024?"
- [ ] Entity resolution — collapse `Apple Inc.` / `AAPL` / `Apple` across filings
- [ ] Sentiment overlay on risk-factor passages
- [ ] Streaming UI with the router's intermediate reasoning visible
- [ ] Page-level citations linking directly to SEC.gov

---

## License

MIT.
