import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest arXiv papers into Qdrant.",
        epilog=(
            "Only --limit is accepted from CLI. "
            "All other ingest settings are read from environment variables or built-in defaults."
        ),
    )

    parser.add_argument("--limit", type=int, default=100, help="Number of papers to ingest")
    return parser.parse_args()
