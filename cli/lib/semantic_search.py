import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "cache/movie_embeddings.npy"


def load_movies(path: str = "data/movies.json") -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["movies"]


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
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