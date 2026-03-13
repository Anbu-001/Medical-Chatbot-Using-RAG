import argparse
from ingestion.ingest_data import ingest
from chatbot.chat_interface import run_cli


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true", help="Ingest documents before starting the chatbot")
    args = parser.parse_args()

    if args.ingest:
        n = ingest()
        print(f"Ingested {n} chunks.")

    run_cli()


if __name__ == "__main__":
    main()
