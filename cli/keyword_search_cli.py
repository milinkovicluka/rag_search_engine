import argparse
import json
import math
import os
import pickle
import string
import sys
from collections import Counter

from nltk.stem import PorterStemmer


punctuation_table = str.maketrans("", "", string.punctuation)
stemmer = PorterStemmer()


def load_stopwords(path: str = "data/stopwords.txt") -> list[str]:
    with open(path) as f:
        raw_words = f.read().splitlines()
        return [w.lower().translate(punctuation_table) for w in raw_words]


STOPWORDS = load_stopwords()


def tokenize_text(text: str) -> list[str]:
    """Lowercase, strip punctuation, split, remove stopwords, and stem."""
    tokens = text.lower().translate(punctuation_table).split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens


def tokenize_term(term: str) -> str:
    """Tokenize a single term. Raises if it doesn't reduce to exactly one token."""
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(f"expected exactly one token, got {len(tokens)}: {tokens}")
    return tokens[0]


def load_movies(path: str = "data/movies.json") -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["movies"]


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            self.docmap[doc_id] = m
            self.__add_document(doc_id, f"{m['title']} {m['description']}")

    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)


def calculate_idf(index: InvertedIndex, token: str) -> float:
    total_docs = len(index.docmap)
    doc_count = len(index.get_documents(token))
    return math.log(total_docs / (doc_count + 1))


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()
    print("Index built and saved to cache/")


def search_command(query: str) -> None:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Error: index not found. Run the 'build' command first.")
        sys.exit(1)

    print(f"Searching for: {query}")

    query_tokens = tokenize_text(query)
    seen_ids = set()
    results = []

    for token in query_tokens:
        doc_ids = index.get_documents(token)
        for doc_id in doc_ids:
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                results.append(doc_id)
                if len(results) >= 5:
                    break
        if len(results) >= 5:
            break

    for i, doc_id in enumerate(results, 1):
        movie = index.docmap[doc_id]
        print(f"{i}. {movie['title']} (ID: {doc_id})")


def tf_command(doc_id: int, term: str) -> None:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Error: index not found. Run the 'build' command first.")
        sys.exit(1)

    token = tokenize_term(term)
    print(index.get_tf(doc_id, token))


def idf_command(term: str) -> None:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Error: index not found. Run the 'build' command first.")
        sys.exit(1)

    token = tokenize_term(term)
    idf = calculate_idf(index, token)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def tfidf_command(doc_id: int, term: str) -> None:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Error: index not found. Run the 'build' command first.")
        sys.exit(1)

    token = tokenize_term(term)
    idf = calculate_idf(index, token)
    tf = index.get_tf(doc_id, token)
    tf_idf = tf * idf

    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to look up")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to look up")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to look up")

    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(args.query)
        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()