import streamlit as st
import sys
import time
sys.path.append("..")

from retrieval.vector_store import vector_search
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import reciprocal_rank_fusion
from reranking.reranker import rerank
from generation.generator import generate_answer
from observability.tracker import RAGTracker

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SEC Filing RAG",
    page_icon="📊",
    layout="wide"
)

# ── Session State ─────────────────────────────────────────────
if "tracker" not in st.session_state:
    st.session_state.tracker = RAGTracker()
if "history" not in st.session_state:
    st.session_state.history = []
if "bm25" not in st.session_state:
    bm25 = BM25Retriever()
    bm25.load_index()
    st.session_state.bm25 = bm25

# ── Header ────────────────────────────────────────────────────
st.title("📊 SEC Filing RAG System")
st.caption(
    "Production RAG over Apple 10-K filings — "
    "Hybrid retrieval · Cross-encoder reranking · Citation enforcement"
)

# ── Sidebar — Metrics ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Retrieval top-k", 10, 30, 20)
    rerank_k = st.slider("Rerank top-k", 3, 10, 5)
    threshold = st.slider("Score threshold", 0.05, 0.30, 0.10, 0.01)

    st.divider()
    st.header("📈 Session Metrics")

    history = st.session_state.history
    if history:
        import numpy as np
        latencies = [h["latency_ms"] for h in history]
        costs = [h["cost_usd"] for h in history]
        citations = [h["citations"] for h in history]
        refusals = sum(1 for h in history if h["refused"])

        col1, col2 = st.columns(2)
        col1.metric("Queries", len(history))
        col2.metric("Refusals", refusals)

        col3, col4 = st.columns(2)
        col3.metric("p50 Latency", f"{np.percentile(latencies,50):.0f}ms")
        col4.metric("p95 Latency", f"{np.percentile(latencies,95):.0f}ms")

        col5, col6 = st.columns(2)
        col5.metric("Avg Citations", f"{np.mean(citations):.1f}")
        col6.metric("Total Cost", f"${sum(costs):.5f}")

        st.divider()
        st.caption("Latency breakdown (avg)")
        for stage in ["retrieval", "reranking", "generation"]:
            stage_lats = [
                h["stages"].get(stage, 0) for h in history
            ]
            if stage_lats:
                st.caption(
                    f"{stage.capitalize()}: "
                    f"{np.mean(stage_lats):.0f}ms"
                )
    else:
        st.caption("Run a query to see metrics")

    st.divider()
    st.header("💡 Example Queries")
    examples = [
        "What are Apple's main risk factors?",
        "What were Apple total net sales in 2025?",
        "What were Apple's operating expenses in 2025?",
        "What were iPhone net sales in 2025?",
        "What regulatory risks does Apple face globally?",
        "What is Apple's plan to acquire Tesla?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.query_input = ex

# ── Main — Query Input ────────────────────────────────────────
query = st.text_input(
    "Ask a question about Apple's SEC filings:",
    value=st.session_state.get("query_input", ""),
    placeholder="e.g. What are Apple's main risk factors?",
    key="query_input"
)

col_run, col_clear = st.columns([1, 5])
run = col_run.button("🔍 Ask", type="primary")
if col_clear.button("🗑️ Clear History"):
    st.session_state.history = []
    st.session_state.tracker = RAGTracker()
    st.rerun()

# ── Main — Pipeline Execution ─────────────────────────────────
if run and query.strip():
    with st.spinner("Running pipeline..."):

        t_start = time.perf_counter()

        # Stage 1: Retrieval
        t0 = time.perf_counter()
        vector_results = vector_search(query, top_k=top_k)
        bm25_results   = st.session_state.bm25.search(query, top_k=top_k)
        merged         = reciprocal_rank_fusion(
                            [vector_results, bm25_results])
        candidates     = merged[:40]
        retrieval_ms   = (time.perf_counter() - t0) * 1000

        # Stage 2: Reranking
        t0 = time.perf_counter()
        reranked    = rerank(query, candidates, top_k=rerank_k)
        reranking_ms = (time.perf_counter() - t0) * 1000

        # Stage 3: Generation
        t0 = time.perf_counter()
        result = generate_answer(
            query=query,
            reranked_chunks=reranked,
            score_threshold=threshold
        )
        generation_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000

    # ── Answer ────────────────────────────────────────────────
    st.divider()
    st.subheader("📝 Answer")

    refused = "not available in the provided filings" in result["answer"].lower()

    if refused:
        st.warning(result["answer"])
    else:
        st.success(result["answer"])

    # ── Metrics Row ───────────────────────────────────────────
    st.divider()
    st.subheader("📊 Query Metrics")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Latency",  f"{total_ms:.0f}ms")
    m2.metric("Retrieval",      f"{retrieval_ms:.0f}ms")
    m3.metric("Reranking",      f"{reranking_ms:.0f}ms")
    m4.metric("Generation",     f"{generation_ms:.0f}ms")
    m5.metric("Citations",      result["citations_count"]
              if "citations_count" in result
              else len(result["citations_in_answer"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Cost",           f"${result['tokens_used'] * 0.00000005:.6f}")
    c2.metric("Tokens Used",    result["tokens_used"])
    c3.metric("Chunks Used",    result["sources_used"])

    # ── Sources ───────────────────────────────────────────────
    if result["citations_in_answer"]:
        st.divider()
        st.subheader("🔗 Citations")
        for cite in result["citations_in_answer"]:
            st.caption(f"• {cite}")

    # ── Retrieved Chunks (expandable) ─────────────────────────
    with st.expander("🔍 View Retrieved Chunks"):
        for i, chunk in enumerate(reranked):
            score = chunk.get("reranker_score", 0)
            meta  = chunk.get("metadata", {})
            st.markdown(
                f"**Chunk {i+1}** | "
                f"Reranker: `{score:.4f}` | "
                f"Source: `{meta.get('ticker')} "
                f"{meta.get('filing_date')}` | "
                f"Index: `{meta.get('chunk_index')}`"
            )
            st.caption(chunk["text"][:400])
            st.divider()

    # ── Save to history ───────────────────────────────────────
    st.session_state.history.append({
        "query":      query,
        "latency_ms": total_ms,
        "cost_usd":   result["tokens_used"] * 0.00000005,
        "citations":  len(result["citations_in_answer"]),
        "refused":    refused,
        "stages": {
            "retrieval":  retrieval_ms,
            "reranking":  reranking_ms,
            "generation": generation_ms
        }
    })

# ── Query History ─────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    st.subheader("🕘 Query History")
    for h in reversed(st.session_state.history[-5:]):
        status = "⛔" if h["refused"] else "✅"
        st.caption(
            f"{status} `{h['query'][:60]}` — "
            f"{h['latency_ms']:.0f}ms · "
            f"{h['citations']} citations · "
            f"${h['cost_usd']:.6f}"
        )
