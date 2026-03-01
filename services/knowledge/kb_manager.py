"""
services/knowledge/kb_manager.py
──────────────────────────────────
ChromaDB knowledge base manager.

Loads documents from the /docs folder into a local vector database.
Each document is split into chunks and embedded using sentence-transformers.
The embedding model runs locally — no external API needed.

ChromaDB persists to disk (./data/chromadb) so documents survive restarts.
You only need to run seed_knowledge_base.py once (or when docs change).
"""
import os
import glob
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions
from logger import get_logger
from config import get_settings

log = get_logger(__name__)
settings = get_settings()

COLLECTION_NAME = "acme_corp_knowledge"
DOCS_DIR = Path(__file__).parent / "docs"

# Using a lightweight local embedding model
# "all-MiniLM-L6-v2" is 22MB, fast, and good for semantic search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class KnowledgeBaseManager:
    """
    Manages the ChromaDB vector store.
    
    Use this to:
    1. Load/seed documents (run once)
    2. Query for relevant context (called per LLM request)
    """

    def __init__(self):
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self._collection = None

    def get_or_create_collection(self):
        """Gets existing collection or creates a new one."""
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"description": "Acme Corp customer support knowledge base"},
        )
        count = self._collection.count()
        log.info("kb_collection_ready", name=COLLECTION_NAME, document_count=count)
        return self._collection

    def seed_from_docs(self, force_reseed: bool = False):
        """
        Loads all .txt files from /docs into ChromaDB.
        Splits each file into chunks of ~200 words for better retrieval.
        
        Args:
            force_reseed: If True, clears existing data and reloads.
        """
        collection = self.get_or_create_collection()

        if collection.count() > 0 and not force_reseed:
            log.info("kb_already_seeded_skipping", count=collection.count())
            return

        if force_reseed:
            log.info("kb_force_reseed_clearing")
            self._client.delete_collection(COLLECTION_NAME)
            collection = self.get_or_create_collection()

        doc_files = glob.glob(str(DOCS_DIR / "*.txt"))
        log.info("kb_seeding_start", files=len(doc_files))

        all_chunks = []
        all_ids = []
        all_metadata = []

        for filepath in doc_files:
            filename = Path(filepath).stem
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = self._chunk_text(content, chunk_size=200)
            log.info("kb_file_chunked", file=filename, chunks=len(chunks))

            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}_chunk_{i}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadata.append({
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                })

        # Add all chunks to ChromaDB in one batch
        collection.add(
            documents=all_chunks,
            ids=all_ids,
            metadatas=all_metadata,
        )

        log.info(
            "kb_seeding_complete",
            total_chunks=len(all_chunks),
            files=len(doc_files),
        )

    def query(self, query_text: str, n_results: int = 2) -> list[dict]:
        """
        Semantic search in the knowledge base.
        
        Args:
            query_text: The caller's question/statement
            n_results:  Number of relevant chunks to return
            
        Returns:
            List of dicts with 'text', 'source', 'distance' keys
        """
        if not self._collection:
            self.get_or_create_collection()

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            hits = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                hits.append({
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "distance": round(dist, 4),
                })

            return hits

        except Exception as e:
            log.error("kb_query_error", error=str(e))
            return []

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 200) -> list[str]:
        """
        Splits text into overlapping chunks of ~chunk_size words.
        Overlap of 20 words ensures context isn't lost at boundaries.
        """
        words = text.split()
        chunks = []
        overlap = 20
        step = chunk_size - overlap

        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks


# Singleton instance
_kb_manager: Optional[KnowledgeBaseManager] = None


def get_kb_manager() -> KnowledgeBaseManager:
    global _kb_manager
    if _kb_manager is None:
        _kb_manager = KnowledgeBaseManager()
        _kb_manager.get_or_create_collection()
    return _kb_manager
