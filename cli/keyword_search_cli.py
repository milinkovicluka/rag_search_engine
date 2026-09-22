import argparse
import json
import string


punctuation_table = str.maketrans("", "", string.punctuation)


def matches_query(query: str, title: str) -> bool:
    """Return True if at least one token in the query appears in any token of the title."""
    query_tokens = query.lower().translate(punctuation_table).split()
    title_tokens = title.lower().translate(punctuation_table).split()
    return any(q_token in t_token for q_token in query_tokens for t_token in title_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    
    match args.command:
        case "search":
            with open("data/movies.json") as data:
                movies = json.load(data)
                results = []
                for movie in movies["movies"]:
                    if matches_query(args.query, movie["title"]):
                        results.append(movie)
            print(f"Searching for: {args.query}")
            for index, item in enumerate(results[:5], 1):
                print(f"{index}. {item['title']}")
        case _:
            parser.print_help()

    
if __name__ == "__main__":
    main()