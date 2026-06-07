# 📊 SEC Filing RAG System

A production-grade **Retrieval Augmented Generation (RAG)** system built over SEC 10-K filings. This project goes beyond tutorial-level RAG by implementing hybrid retrieval, cross-encoder reranking, citation enforcement, iXBRL parsing, and full observability — the same patterns used in production AI systems at scale.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-red)
![Groq](https://img.shields.io/badge/LLM-Groq_Llama3-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Why This Project

Most RAG tutorials stop at "embed documents, search, generate." That works for demos but fails in production because:

- Embedding similarity alone misses exact keyword matches (ticker names, dollar amounts)
- No way to know if the answer is grounded or hallucinated
- No visibility into latency, cost, or quality per query
- SEC filings are iXBRL — not plain text — requiring a custom parser

This project solves all four problems with a layered architecture that mirrors what AI engineers build at real companies.

---

## 🏗️ Architecture

```
SEC EDGAR API (free, no key required)
        ↓
Document Ingestion
├── iXBRL Parser (extracts $416.2B not raw "416161")
└── Semantic Chunker (512 tokens, 64 overlap)
        ↓
┌─────────────────────────────────────────────┐
│           Hybrid Retrieval                  │
│  BM25 (keyword) + Vector (semantic) → RRF   │
└─────────────────────────────────────────────┘
        ↓
Cross-Encoder Reranking (BGE)
        ↓
LLM Generation + Citation Enforcement (Groq)
        ↓
Cited, Grounded Answer
        ↑
Observability Layer (tracks every query)
├── Latency per stage (retrieval/reranking/generation)
├── Cost per query (token-level)
└── Quality metrics (citations, refusal rate, reranker score)
```

---

## ⚡ Performance

| Metric | Value |
|---|---|
| Retrieval p50 latency | 14ms |
| Reranking p50 latency | 442ms |
| End-to-end p50 latency | ~1,000ms |
| Cost per query | $0.000058 |
| Avg citations per answer | 3.3 |
| Hallucinations in testing | 0 |
| Correct refusal rate | 1/6 (Tesla query) |
| Reranker avg confidence | 0.923 |

---

## 🔑 Key Design Decisions

### 1. Hybrid Retrieval — BM25 + Vector Search

**What we use:** BM25Okapi (rank-bm25) + Qdrant vector search, merged via Reciprocal Rank Fusion (RRF)

**Why not vector search alone?**

Pure vector search uses embedding similarity — great for semantic meaning but weak on exact matches. When a user asks about "AAPL Q4 revenue" or "$416.2B", the embedding model may not rank exact matches at the top. BM25 is a classical keyword algorithm that excels at exact term matching.

**Why not BM25 alone?**

BM25 fails on paraphrasing and intent. "What risks could hurt Apple's stock?" won't match chunks that say "material adverse effect on financial condition" — even though they're the same concept.

**Why RRF for merging?**

Reciprocal Rank Fusion combines ranked lists without needing to tune weights. It's robust, parameter-free, and industry-standard. Weighted averaging requires manual tuning and breaks when score distributions shift.

```
BM25:   finds "iPhone net sales" chunks (exact keywords)   ✅
Vector: finds "revenue from smartphone segment" chunks     ✅
RRF:    combines both, rewards chunks appearing in both    ✅
```

---

### 2. Cross-Encoder Reranking — Why Not Just Use Retrieval Scores?

**What we use:** BAAI/bge-reranker-base (CrossEncoder)

**The problem with retrieval scores:**

Both BM25 and vector search score documents independently — they don't compare the query and document together. A chunk can score high because it shares vocabulary with the query, not because it actually answers it.

**What the reranker does differently:**

A cross-encoder reads the query and document together as a pair, producing a single relevance score. This is much more accurate but slower — which is why we run it only on the top 40 candidates from retrieval, not all chunks.

```
Retrieval (fast, recall-focused):
  Query → score each chunk independently → top 40

Reranking (slower, precision-focused):
  [Query + Chunk 1] → score
  [Query + Chunk 2] → score
  ...
  Sort by joint score → top 5
```

**Real example from testing:**

Before reranking: chunk about "seasonal holiday demand" ranked #2 for risk factors query
After reranking: correctly pushed to bottom, cybersecurity risk chunk moved to #1

---

### 3. iXBRL Parser — Why Not PyPDF or LlamaParse?

**What we use:** Custom BeautifulSoup parser with iXBRL tag handling

**The problem:**

SEC 10-K filings are not plain PDFs — they are Inline XBRL (iXBRL), a hybrid HTML/XML format where financial numbers are stored as raw scaled integers with metadata:

```html
<ix:nonfraction name="us-gaap:Revenues" scale="6" unitref="usd">
  416161
</ix:nonfraction>
```

Standard PDF parsers and naive HTML parsers extract `416161` — a meaningless number. Our parser applies the scale factor to produce `$416.2B` — the actual revenue figure.

**Why not LlamaParse?**

LlamaParse is a paid API with per-page pricing. For a portfolio project processing multiple filings across multiple companies, cost adds up. Our custom parser is free, handles iXBRL correctly, and gives full control over extraction logic.

**Why not PyMuPDF?**

SEC EDGAR serves 10-K filings as HTML/iXBRL, not PDF. PyMuPDF is a PDF library and would require downloading the PDF version separately, which is harder to automate via the EDGAR API.

---

### 4. Citation Enforcement — Why It Matters

**What we use:** Structured system prompt with strict citation rules + regex extraction

**The problem with standard RAG:**

Standard RAG generates an answer from retrieved chunks but doesn't force the model to say where each claim came from. In financial and legal domains, an uncited answer is unacceptable — you can't verify it.

**How we enforce citations:**

The system prompt contains strict rules:
- Every factual claim must include `[Source: TICKER 10-K DATE, chunk N]`
- If the answer is not in the context, return a specific refusal phrase
- Never use information not present in the provided chunks

**Hallucination test result:**

```
Query:  "What is Apple's plan to acquire Tesla?"
Answer: "This information is not available in the provided filings."
Citations: 0
```

The system correctly refused rather than hallucinating an acquisition plan.

---

### 5. Observability — Why Most RAG Projects Skip This

**What we use:** Custom RAGTracker class with JSONL logging

**Why it matters:**

Without observability, you have no idea:
- Which pipeline stage is slow (retrieval? reranking? generation?)
- How much each query costs
- Whether answer quality is degrading over time
- What percentage of queries are being refused

**What we track per query:**

```python
{
  "query": "What are Apple's main risk factors?",
  "stages": {
    "retrieval":   {"latency_ms": 14},
    "reranking":   {"latency_ms": 442},
    "generation":  {"latency_ms": 611}
  },
  "metrics": {
    "vector_hits": 20,
    "bm25_hits": 20,
    "reranker_top_score": 1.0,
    "citations_count": 6,
    "cost_usd": 0.000058,
    "answer_refused": false
  }
}
```

Every query is logged to `data/observability_log.jsonl` for offline analysis.

---

### 6. Embedding Model — Why BAAI/bge-small-en-v1.5

**Alternatives considered:**

| Model | Size | Cost | Quality | Decision |
|---|---|---|---|---|
| OpenAI text-embedding-3-small | API | $$ per call | High | ❌ Paid API |
| all-MiniLM-L6-v2 | 80MB | Free | Medium | ❌ Weaker on financial text |
| BAAI/bge-small-en-v1.5 | 130MB | Free | High | ✅ Chosen |
| BAAI/bge-large-en-v1.5 | 1.3GB | Free | Highest | ❌ Too slow for dev |

BGE-small scores near the top of MTEB benchmarks for its size class, runs locally with no API cost, and handles financial terminology well.

---

### 7. LLM — Why Groq Instead of OpenAI

**What we use:** Groq API with Llama 3.1 8B Instant

**Why not OpenAI GPT-4o?**

GPT-4o requires a paid API key. For a portfolio project meant to be reproducible by anyone, a free tier option is more accessible.

**Why Groq over Ollama (local)?**

Ollama runs entirely locally — great for privacy but requires 8GB+ RAM and is slow on CPU. Groq provides free API access with extremely fast inference (their custom LPU hardware). For a Colab-based project, Groq gives production-like speed without local hardware requirements.

**Why Groq over Gemini?**

Groq's API is OpenAI-compatible — changing from OpenAI to Groq required changing only 3 lines of code. Gemini uses a different SDK requiring more refactoring.

---

### 8. Vector Database — Why Qdrant

**Alternatives considered:**

| DB | Hosting | Cost | Local Mode | Decision |
|---|---|---|---|---|
| Pinecone | Cloud only | $$ | ❌ | ❌ Paid |
| Weaviate | Cloud/Local | Free tier | ✅ | ❌ Heavier setup |
| ChromaDB | Local | Free | ✅ | ✅ Also good |
| Qdrant | Cloud/Local | Free | ✅ | ✅ Chosen |

Qdrant was chosen for its local file-based mode (`QdrantClient(path="data/qdrant_db")`), which requires zero server setup, and its clean Python API. It can be swapped to Qdrant Cloud for production with one line change.

---

## 📁 Project Structure

```
sec-rag-system/
├── ingestion/
│   ├── edgar_fetcher.py      # EDGAR API client — fetches 10-K filings
│   ├── parser.py             # iXBRL parser — extracts financial figures
│   └── chunker.py            # Semantic chunker with noise filtering
├── retrieval/
│   ├── vector_store.py       # Qdrant operations + BGE embeddings
│   ├── bm25_retriever.py     # BM25 keyword search + pickle persistence
│   └── hybrid_retriever.py   # RRF fusion of BM25 + vector results
├── reranking/
│   └── reranker.py           # BGE cross-encoder reranking
├── generation/
│   └── generator.py          # Groq LLM + citation enforcement
├── observability/
│   └── tracker.py            # Per-query latency/cost/quality tracking
├── app/
│   ├── gradio_app.py         # Gradio UI (Colab demo)
│   └── streamlit_app.py      # Streamlit UI (GitHub deployment)
├── data/
│   ├── raw/                  # Downloaded 10-K HTML files
│   ├── qdrant_db/            # Local vector database
│   └── bm25_index.pkl        # Serialized BM25 index
├── rag_pipeline.py           # End-to-end pipeline + ask() function
├── .env.example              # Environment variable template
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/PavanKAgnihotri/SEC-RAG-System.git
cd SEC-RAG-System
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add:
# EDGAR_USER_AGENT=FirstName LastName email@example.com
# GROQ_API_KEY=your_groq_key (free at console.groq.com)
```

### 3. Build the pipeline
```bash
python rag_pipeline.py
# Downloads AAPL 10-K, parses, chunks, embeds, indexes
# Takes ~2 minutes on first run
```

### 4. Launch UI
```bash
# Streamlit
streamlit run app/streamlit_app.py

# Gradio
python app/gradio_app.py
```

---

## 💬 Example Queries

```
✅ "What are Apple's main risk factors?"
✅ "What were Apple total net sales in 2025?"
✅ "What were Apple's operating expenses in 2025?"
✅ "What were iPhone net sales in 2025?"
✅ "What regulatory risks does Apple face globally?"
⛔ "What is Apple's plan to acquire Tesla?"
   → "This information is not available in the provided filings."
```

---

## ⚠️ Known Limitations

**Reranker terminology mismatch**
The BGE reranker interprets "cash position" as a financial hedging term rather than a balance sheet item. Workaround: use "cash and cash equivalents". Fix: fine-tune the reranker on SEC-specific Q&A pairs using FinBERT or a domain-adapted model.

**Table percentage columns**
Financial tables with percentage change columns create cell fragmentation during extraction. The numbers are correct but the raw chunk text is noisy (e.g. `62,151 | 8 | %`).

**Groq free tier rate limits**
The free tier throttles under sustained load causing generation latency spikes up to 15-20 seconds. Mitigation: add `time.sleep(3)` between batch queries. Fix: upgrade to paid tier or self-host Llama 3 via Ollama.

**Single company, single filing**
Currently indexes one 10-K filing per run. Multi-company and multi-year comparison is a planned enhancement.

---

## 🔮 Future Work

- [ ] RAGAS eval pipeline integrated into GitHub Actions CI
- [ ] Multi-company comparison (AAPL vs MSFT vs GOOGL)
- [ ] Earnings call transcript ingestion (via Motley Fool / Seeking Alpha)
- [ ] Fine-tuned financial reranker (FinBERT-based)
- [ ] Qdrant Cloud deployment for persistent storage
- [ ] Streaming responses in UI
- [ ] Query history and session persistence

---

## 🛠️ Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Data source | SEC EDGAR API | Free, public, no scraping |
| HTML parsing | BeautifulSoup | iXBRL tag handling |
| Chunking | LangChain RecursiveCharacterTextSplitter | Semantic boundary awareness |
| Vector DB | Qdrant | Free local mode, cloud-ready |
| Embeddings | BAAI/bge-small-en-v1.5 | Free, strong MTEB score |
| Keyword search | rank-bm25 | Lightweight, no infra needed |
| Reranker | BAAI/bge-reranker-base | Free cross-encoder, strong precision |
| LLM | Groq / Llama 3.1 8B | Free tier, OpenAI-compatible |
| UI (demo) | Gradio | Zero-config Colab deployment |
| UI (portfolio) | Streamlit | Clean, GitHub-deployable |
| Observability | Custom RAGTracker | Per-stage latency + cost + quality |

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

Built following the production RAG patterns outlined by Aishwarya Srinivasan's AI Engineer project guide. SEC financial data sourced from EDGAR, a service of the U.S. Securities and Exchange Commission.
