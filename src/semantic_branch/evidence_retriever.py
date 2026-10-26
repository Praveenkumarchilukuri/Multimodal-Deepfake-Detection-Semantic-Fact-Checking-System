"""
Hybrid Evidence Retriever Module
Retrieves candidate factual evidence from indexed knowledge bases
and real-time verification sources.
"""

from typing import List, Optional
from .vector_store import VectorKnowledgeStore, EvidenceChunk
from .claim_extractor import AtomicClaim


class EvidenceRetriever:
    """
    Coordinates evidence retrieval for atomic claims across local and external sources.
    """

    def __init__(self, vector_store: VectorKnowledgeStore, top_k: int = 3):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve_for_claim(self, claim: AtomicClaim) -> List[EvidenceChunk]:
        """
        Retrieves top_k evidence chunks matching the claim text.
        """
        results = self.vector_store.query(claim.text, top_k=self.top_k)
        return results

    def retrieve_batch(self, claims: List[AtomicClaim]) -> List[List[EvidenceChunk]]:
        """
        Batch retrieval for multiple atomic claims.
        """
        return [self.retrieve_for_claim(c) for c in claims]
