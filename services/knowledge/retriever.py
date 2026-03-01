"""
services/knowledge/retriever.py
─────────────────────────────────
RAG Retriever — fetches relevant knowledge base context before each LLM call.

This is what makes the AI agent "know" about company-specific information
without hallucinating. Before sending the user's message to the LLM,
we search the knowledge base and inject relevant context into the system prompt.

Relevance threshold:
  ChromaDB returns a "distance" score (lower = more similar).
  We only inject context if distance < 0.7 (reasonably relevant).
  Above that threshold, we trust the LLM's general knowledge instead.
"""
from typing import Optional
from logger import get_logger
from services.knowledge.kb_manager import get_kb_manager
from services.metrics.collector import kb_queries_total, kb_hits_total

log = get_logger(__name__)

# Cosine distance threshold — below this = relevant enough to inject
RELEVANCE_THRESHOLD = 0.7


class KnowledgeRetriever:
    """
    Called by VoiceAIPipeline before each LLM request.
    Returns formatted context string if relevant docs found, else None.
    """

    def __init__(self):
        self._kb = get_kb_manager()

    async def retrieve(self, query: str) -> Optional[str]:
        """
        Searches the knowledge base for context relevant to the caller's query.
        
        Args:
            query: The caller's transcribed speech
            
        Returns:
            Formatted context string for injection into LLM system prompt,
            or None if no relevant context found.
        """
        kb_queries_total.inc()

        if not query.strip():
            return None

        hits = self._kb.query(query_text=query, n_results=2)

        if not hits:
            log.debug("kb_no_results", query=query[:80])
            return None

        # Filter by relevance threshold
        relevant_hits = [h for h in hits if h["distance"] < RELEVANCE_THRESHOLD]

        if not relevant_hits:
            log.debug(
                "kb_results_below_threshold",
                query=query[:80],
                best_distance=hits[0]["distance"] if hits else "N/A",
            )
            return None

        kb_hits_total.inc()

        # Format for injection into LLM prompt
        context_parts = []
        for hit in relevant_hits:
            context_parts.append(
                f"[Source: {hit['source']}]\n{hit['text']}"
            )

        formatted_context = "\n\n".join(context_parts)

        log.info(
            "kb_context_retrieved",
            query=query[:80],
            hits=len(relevant_hits),
            sources=[h["source"] for h in relevant_hits],
            best_distance=relevant_hits[0]["distance"],
        )

        return formatted_context
