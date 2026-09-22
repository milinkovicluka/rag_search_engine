import argparse
import json
import string

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    punctuation_table = str.maketrans("", "", string.punctuation)

    match args.command:
        case "search":
            with open("data/movies.json") as data:
                movies = json.load(data)
                results = []
                for movie in movies["movies"]:
                    clean_query = args.query.translate(punctuation_table)
                    clean_title = movie["title"].translate(punctuation_table)
                    if clean_query.lower() in clean_title.lower():
                        results.append(movie)
            print(f"Searching for: {args.query}")
            for index, item in enumerate(results[:5], 1):
                print(f"{index}. {item['title']}")
        case _:
            parser.print_help()

    

    
if __name__ == "__main__":
    main()