import json
import os
import re
from typing import TypedDict

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "cache/movie_embeddings.npy"
CHUNK_EMBEDDINGS_PATH = "cache/chunk_embeddings.npy"
CHUNK_METADATA_PATH = "cache/chunk_metadata.json"


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


def load_movies(path: str = "data/movies.json") -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["movies"]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    step = chunk_size - overlap

    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))

        if i + chunk_size >= len(words):
            break

        i += step

    return chunks


def semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    if overlap >= max_chunk_size:
        raise ValueError("overlap must be smaller than max_chunk_size")

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    step = max_chunk_size - overlap

    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sentences = sentences[i:i + max_chunk_size]
        chunks.append(" ".join(chunk_sentences))

        if i + max_chunk_size >= len(sentences):
            break

        i += step

    return chunks


class SemanticSearch:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("text must not be empty or whitespace-only")

        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        texts = [f"{doc['title']}: {doc['description']}" for doc in documents]

        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        scored = []
        for doc, doc_embedding in zip(self.documents, self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)

        results = []
        for score, doc in scored[:limit]:
            results.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            })

        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks: list[str] = []
        chunk_metadata: list[ChunkMetadata] = []

        for movie_idx, doc in enumerate(documents):
            description = doc.get("description", "")
            if not description or not description.strip():
                continue

            chunks = semantic_chunk_text(description, max_chunk_size=4, overlap=1)
            total_chunks = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)

        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH) as f:
                data = json.load(f)
            self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)


def verify_model() -> None:
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings() -> None:
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def semantic_search_command(query: str, limit: int = 5) -> None:
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_embeddings(documents)

    results = semantic_search.search(query, limit)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"  {result['description'][:100]}...")
        print()


def chunk_command(text: str, chunk_size: int = 200, overlap: int = 0) -> None:
    chunks = chunk_text(text, chunk_size, overlap)

    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def semantic_chunk_command(text: str, max_chunk_size: int = 4, overlap: int = 0) -> None:
    chunks = semantic_chunk_text(text, max_chunk_size, overlap)

    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def embed_chunks_command() -> None:
    documents = load_movies()
    chunked_search = ChunkedSemanticSearch()
    embeddings = chunked_search.load_or_create_chunk_embeddings(documents)

    print(f"Generated {len(embeddings)} chunked embeddings")