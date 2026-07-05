"""
Builds the Chroma vector store for the Yoga & Asana wisdom chatbot from the
scraped Asanas & Mudras article data in ../scraper/output.

Usage:
    python ingest.py             # build (or rebuild) the vector store
    python ingest.py --rebuild   # wipe and rebuild from scratch
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).parent
ARTICLES_PATH = BASE_DIR.parent / "scraper" / "output" / "asana_mudra_articles.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "yoga_asana_mudra"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_articles() -> list[dict]:
    if not ARTICLES_PATH.exists():
        raise FileNotFoundError(
            f"Could not find scraped article data at {ARTICLES_PATH}. "
            "Run the scraper first (see ../scraper/README.md)."
        )
    with ARTICLES_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def articles_to_documents(articles: list[dict]) -> list[Document]:
    documents = []
    for article in articles:
        text = f"{article['title']}\n\n{article['content']}"
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "id": article["id"],
                    "title": article["title"],
                    "url": article["url"],
                    "published": article["published"],
                    "tags": ", ".join(article.get("tags", [])),
                },
            )
        )
    return documents


def build_vector_store(rebuild: bool) -> None:
    if rebuild and CHROMA_DIR.exists():
        print(f"Removing existing vector store at {CHROMA_DIR}...")
        shutil.rmtree(CHROMA_DIR)

    print("Loading scraped articles...")
    articles = load_articles()
    print(f"Loaded {len(articles)} articles.")

    documents = articles_to_documents(articles)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads it)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Embedding and storing chunks in Chroma at {CHROMA_DIR}...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print("Done. Vector store is ready.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Wipe and rebuild the vector store from scratch")
    return parser.parse_args(argv)


if __name__ == "__main__":
    import sys

    args = parse_args(sys.argv[1:])
    build_vector_store(rebuild=args.rebuild)
