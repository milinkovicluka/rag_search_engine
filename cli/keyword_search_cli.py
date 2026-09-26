import argparse
import json
import os
import pickle
import string

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


def matches_query(query: str, title: str) -> bool:
    """Return True if at least one processed query token appears in any processed title token."""
    query_tokens = tokenize_text(query)
    title_tokens = tokenize_text(title)
    return any(q_token in t_token for q_token in query_tokens for t_token in title_tokens)


def load_movies(path: str = "data/movies.json") -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["movies"]


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

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


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()
    docs = index.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            movies = load_movies()
            results = []
            for movie in movies:
                if matches_query(args.query, movie["title"]):
                    results.append(movie)
            print(f"Searching for: {args.query}")
            for index, item in enumerate(results[:5], 1):
                print(f"{index}. {item['title']}")
        case "build":
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()