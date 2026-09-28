import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    semantic_search_command,
    chunk_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model loads correctly")

    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a text input")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Build or load movie embeddings and verify them")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate an embedding for a query string")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search movies by meaning using semantic similarity")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into fixed-size pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            semantic_search_command(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()