# RAG Search Engine

**A movie search engine that goes from basic keyword matching all the way to LLM-generated answers and image-based search.**

![Demo of hybrid search and conversational Q&A](docs/demo.gif)

Search a dataset of 5,000 movies by keywords, by meaning, or by uploading a picture, then get a natural-language answer grounded in the results. Built step by step as part of the [Boot.dev](https://www.boot.dev) RAG course.

## Motivation

Every app now has an "AI search" box, and I kept wondering what was actually happening behind it. I tried building one the quick way, by wiring a vector database and an LLM together with a framework, and it *worked*... until it didn't. When it returned the wrong movie, I had no idea whether the problem was the query, the embeddings, the ranking, or the LLM, because every layer was a black box. So I built the whole pipeline from scratch, one layer at a time, starting with a plain inverted index and ending with image search. Now when a search fails, I can trace it stage by stage, measure the fix with real precision and recall numbers, and explain *why* it works.

Along the way, a few questions shaped the design:

- **Why pick between keyword and semantic search** when each one catches what the other misses?
- **Why trust the first ranking** when a cross-encoder or an LLM can double-check it?
- **Why guess whether a change helped** when a golden dataset can tell you?
- **Why should a query have to be text** when images and words can share the same vector space?

## Quick Start

You'll need **Python 3.12+**, [**uv**](https://docs.astral.sh/uv/), and a free [**OpenRouter**](https://openrouter.ai) API key.

```bash
git clone https://github.com/<your-username>/rag_search_engine.git
cd rag_search_engine
uv sync
echo "OPENROUTER_API_KEY=your_key_here" > .env
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears"
```

> **Note:** The first run downloads embedding models and builds the search index, so it can take a few minutes. Everything is cached in `cache/` after that.

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
# Weighted: alpha controls BM25 vs. semantic (1.0 = all keyword, 0.0 = all semantic)
uv run cli/hybrid_search_cli.py weighted-search "bear movie" --alpha 0.5 --limit 10

# Reciprocal Rank Fusion
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears" --limit 10
```

#### Query enhancement (`--enhance`)

| Mode | What it does |
|------|--------------|
| `spell` | Fixes typos only (`"scray movei"` → `"scary movie"`) |
| `rewrite` | Turns vague queries into specific ones (`"that bear movie with leo"` → `"The Revenant Leonardo DiCaprio bear attack"`) |
| `expand` | Appends synonyms and related terms |

```bash
uv run cli/hybrid_search_cli.py rrf-search "scray bear movei" --enhance spell
```

#### Re-ranking (`--rerank-method`)

| Method | Speed | Notes |
|--------|-------|-------|
| `individual` | 🐢 Slow | LLM scores each result separately, one API call per result |
| `batch` | ⚡ Medium | LLM ranks all results in a single call |
| `cross_encoder` | 🚀 Fast | Runs locally, no API calls |

```bash
uv run cli/hybrid_search_cli.py rrf-search "bear movie" --rerank-method cross_encoder
```

#### LLM-as-judge (`--evaluate`)

```bash
uv run cli/hybrid_search_cli.py rrf-search "dinosaur park" --evaluate
```

### Evaluation

Scores the search against `data/golden_dataset.json` and prints precision, recall, and F1 for each test query.

```bash
uv run cli/evaluation_cli.py --limit 5
```

### Retrieval-Augmented Generation

```bash
uv run cli/augmented_generation_cli.py rag "movies about action and dinosaurs"
uv run cli/augmented_generation_cli.py summarize "space adventures" --limit 5
uv run cli/augmented_generation_cli.py citations "space adventures"
uv run cli/augmented_generation_cli.py question "what's a good family movie?"
```

### Multimodal search

```bash
# Rewrite a text query using an image
uv run cli/describe_image_cli.py --image data/paddington.jpeg --query "bear movie"

# Find movies that match an image
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

## Contributing

Contributions are welcome! Ideas I'd love help with:

- A **web UI** (Streamlit, Gradio, or FastAPI)
- Swapping NumPy for a **vector database** like Chroma or Qdrant
- More test cases in the **golden dataset**

### How to contribute

1. **Fork** the repo and clone your fork
2. Create a branch: `git checkout -b my-feature`
3. Make your changes and run the evaluation to check that search quality didn't drop:
   ```bash
   uv run cli/evaluation_cli.py --limit 5
   ```
4. Commit, push, and **open a pull request** describing what you changed and why

Found a bug or have an idea? [Open an issue](https://github.com/<your-username>/rag_search_engine/issues).
