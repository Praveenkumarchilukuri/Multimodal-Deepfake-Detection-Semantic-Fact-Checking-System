"""
Unit Tests for Semantic Branch (Claim Extractor, Vector Store, NLI Verifier)
"""

import os
import pytest
from src.semantic_branch.claim_extractor import ClaimExtractor
from src.semantic_branch.whisper_transcriber import TranscriptionResult, TimestampedSegment
from src.semantic_branch.vector_store import VectorKnowledgeStore, DenseEmbeddingEncoder
from src.semantic_branch.nli_verifier import NLIVerifier


def test_claim_extractor_filtering():
    extractor = ClaimExtractor(max_claims=3)
    transcript = TranscriptionResult(
        full_text="Hello everyone. The European Central Bank is located in Frankfurt. Also, did you know it was founded recently?",
        segments=[
            TimestampedSegment(0.0, 3.0, "Hello everyone."),
            TimestampedSegment(3.0, 6.0, "The European Central Bank is located in Frankfurt."),
            TimestampedSegment(6.0, 9.0, "Also, did you know it was founded recently?")
        ],
        language="en",
        duration_sec=9.0,
        has_speech=True
    )

    claims = extractor.extract_claims(transcript)
    assert len(claims) >= 1
    # Check that the factual assertion is captured and greeting is pruned
    assert "European Central Bank" in claims[0].text


def test_vector_store_retrieval():
    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "knowledge_base", "fact_checks_corpus.json"
    )
    store = VectorKnowledgeStore(corpus_path=corpus_path)

    results = store.query("Zelensky surrendered in March 2022", top_k=2)
    assert len(results) > 0
    assert results[0].ground_truth_verdict == "FALSE"
    assert "Zelensky" in results[0].claim_reference


def test_nli_verifier_contradiction():
    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "knowledge_base", "fact_checks_corpus.json"
    )
    store = VectorKnowledgeStore(corpus_path=corpus_path)
    verifier = NLIVerifier(device="cpu")

    # Claim that contradicts ground truth
    from src.semantic_branch.claim_extractor import AtomicClaim
    false_claim = AtomicClaim(
        claim_id="c1",
        text="Volodymyr Zelensky announced surrender to Russian forces.",
        start_sec=0.0,
        end_sec=3.0,
        confidence=0.9
    )
    evidence = store.query(false_claim.text, top_k=1)

    result = verifier.verify_claims([false_claim], [evidence])
    assert result.s_sem >= 0.70  # High inconsistency score
    assert result.num_contradictions == 1
