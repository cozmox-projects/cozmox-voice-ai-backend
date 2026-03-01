#!/usr/bin/env python3
"""
scripts/seed_knowledge_base.py
────────────────────────────────
One-time setup script to load knowledge documents into ChromaDB.

Run this once before starting the system:
    python scripts/seed_knowledge_base.py

Run with --force to reload if you update the docs:
    python scripts/seed_knowledge_base.py --force
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from services.knowledge.kb_manager import KnowledgeBaseManager
from logger import get_logger

log = get_logger("seed_kb")


def main():
    parser = argparse.ArgumentParser(description="Seed the knowledge base with documents")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear existing data and reload all documents",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Voice AI Agent — Knowledge Base Seeder")
    print("=" * 60)

    kb = KnowledgeBaseManager()
    kb.seed_from_docs(force_reseed=args.force)

    # Test a few queries to verify it's working
    print("\n--- Running verification queries ---\n")

    test_queries = [
        "What is your refund policy?",
        "How long does it take to get a refund?",
        "What payment methods do you accept?",
        "The product is too expensive",
        "I want to cancel my subscription",
        "How do I track my order?",
    ]

    for query in test_queries:
        results = kb.query(query_text=query, n_results=1)
        if results:
            hit = results[0]
            print(f"Query: '{query}'")
            print(f"  → Source: {hit['source']} (distance: {hit['distance']})")
            print(f"  → Preview: {hit['text'][:100]}...")
            print()
        else:
            print(f"Query: '{query}' → NO RESULTS\n")

    print("✅ Knowledge base ready!")


if __name__ == "__main__":
    main()
