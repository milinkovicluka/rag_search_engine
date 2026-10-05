# RAG Search Engine

**A movie search engine that goes from basic keyword matching all the way to LLM-generated answers and image-based search.**

Search a dataset of 5,000 movies by keywords, by meaning, or by uploading a picture, then get a natural-language answer grounded in the results. Built step by step as part of the [Boot.dev](https://www.boot.dev) RAG course.

## Why This Exists

Most search bars fail in one of two ways: they miss results because you didn't type the *exact* word, or they return vaguely related junk. This project explores how real retrieval systems fix both problems by **layering techniques**:

1. **Keyword search** catches exact matches
2. **Semantic search** catches meaning ("scary bear movie" → "terrifying grizzly attack")
3. **Hybrid search** combines both so neither blind spot wins
4. **LLMs** clean up queries, re-rank results, and write the final answer


## Motivation
Every app now has an "AI search" box, and I kept wondering what was actually happening behind it. I tried building one the quick way, by wiring a vector database and an LLM together with a framework, and it worked... until it didn't. When it returned the wrong movie, I had no idea whether the problem was the query, the embeddings, the ranking, or the LLM, because every layer was a black box. So I built the whole pipeline from scratch, one layer at a time, starting with a plain inverted index and ending with image search. Now when a search fails, I can trace it stage by stage, measure the fix with real precision and recall numbers, and explain why it works.
Along the way, a few questions shaped the design:
Why pick between keyword and semantic search when each one catches what the other misses?
Why trust the first ranking when a cross-encoder or an LLM can double-check it?
Why guess whether a change helped when a golden dataset can tell you?
Why should a query have to be text when images and words can share the same vector space?


## Features

### 🔤 Keyword Search
- Tokenization, stopword removal, and Porter stemming
- Inverted index with on-disk caching
- **BM25** scoring with term-frequency saturation and document-length normalization

### 🧠 Semantic Search
- Sentence embeddings with `all-MiniLM-L6-v2` (384 dimensions)
- Cosine similarity ranking
- Fixed-size and sentence-based **chunking** with overlap

### 🔀 Hybrid Search
- Weighted scoring with min-max normalization
- **Reciprocal Rank Fusion (RRF)**

### ✨ LLM Enhancements
- **Query enhancement:** spelling correction, rewriting, and expansion
- **Re-ranking:** per-document LLM scoring, batch LLM ranking, or a local cross-encoder

### 📊 Evaluation
- **Precision@K**, **Recall@K**, and **F1** against a golden dataset
- **LLM-as-judge** relevance scoring (0–3)

### 💬 Retrieval-Augmented Generation
- Answer questions using retrieved movies as context
- Multi-document **summaries**
- Answers with **citations** (`[1]`, `[2]`, …)
- Casual conversational Q&A

### 🖼️ Multimodal Search
- Rewrite a text query using an image
- Search movies **with an image** using CLIP embeddings

## Getting Started

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- An [OpenRouter](https://openrouter.ai) API key (for LLM features)

### Installation

```bash
git clone <your-repo-url>
cd rag_search_engine
uv sync
```

Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=your_key_here
```

> **Note:** The first run downloads embedding models and builds the search index, so it can take a few minutes. Results are cached in `cache/` after that.

## Usage

### Keyword search
```bash
uv run cli/keyword_search_cli.py build
uv run cli/keyword_search_cli.py bm25search "bear attack"
```

### Semantic search
```bash
uv run cli/semantic_search_cli.py search "a cute bear in london"
```

### Hybrid search
```bash
# Reciprocal Rank Fusion
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears" --limit 10

# With query enhancement and re-ranking
uv run cli/hybrid_search_cli.py rrf-search "scray bear movei" --enhance spell --rerank-method cross_encoder

# Rate the results with an LLM judge
uv run cli/hybrid_search_cli.py rrf-search "dinosaur park" --evaluate
```

### Evaluation
```bash
uv run cli/evaluation_cli.py --limit 5
```

### RAG
```bash
uv run cli/augmented_generation_cli.py rag "movies about action and dinosaurs"
uv run cli/augmented_generation_cli.py summarize "space adventures"
uv run cli/augmented_generation_cli.py citations "space adventures"
uv run cli/augmented_generation_cli.py question "what's a good family movie?"
```

### Multimodal
```bash
uv run cli/describe_image_cli.py --image data/paddington.jpeg --query "bear movie"
uv run cli/multimodal_search_cli.py image_search data/paddington.jpeg
```

## Project Structure

```
rag_search_engine/
├── cli/
│   ├── keyword_search_cli.py        # BM25 + inverted index
│   ├── semantic_search_cli.py       # Embedding search & chunking
│   ├── hybrid_search_cli.py         # Weighted / RRF search
│   ├── evaluation_cli.py            # Precision, recall, F1
│   ├── augmented_generation_cli.py  # RAG commands
│   ├── describe_image_cli.py        # Image-based query rewriting
│   ├── multimodal_search_cli.py     # CLIP image search
│   └── lib/
│       ├── semantic_search.py
│       ├── hybrid_search.py
│       ├── multimodal_search.py
│       └── search_utils.py
├── data/                            # Movie dataset, golden dataset, images
└── cache/                           # Generated index & embeddings
```

## Tech Stack

- **Python** with **uv**
- **NLTK** for stemming
- **sentence-transformers** for embeddings, cross-encoders, and CLIP
- **NumPy** for vector math
- **OpenRouter** (via the `openai` SDK) for LLM calls

## Known Limitations

- Models and indexes reload on every CLI run
- Embeddings live in NumPy arrays, which is fine for thousands of documents but not millions
- The free OpenRouter model is slow and sometimes inconsistent, so LLM-based features can vary between runs
