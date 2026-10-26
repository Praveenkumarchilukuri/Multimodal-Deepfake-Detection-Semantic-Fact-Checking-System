"""
Atomic Claim Extraction Module
Parses spoken transcripts and breaks multi-sentence spoken discourse
into individual, testable atomic factual claims.
"""

from dataclasses import dataclass
from typing import List, Optional
import re

from .whisper_transcriber import TranscriptionResult, TimestampedSegment


@dataclass
class AtomicClaim:
    claim_id: str
    text: str
    start_sec: float
    end_sec: float
    confidence: float


class ClaimExtractor:
    """
    Decomposes transcripts into verifiable propositions.
    Filters out conversational discourse markers, pleasantries, and questions.
    """

    def __init__(self, max_claims: int = 5):
        self.max_claims = max_claims
        # Common discourse fillers to prune
        self.fillers = re.compile(
            r"^(well,|so,|you know,|um,|uh,|actually,|look,|i mean,)\s*",
            re.IGNORECASE
        )

    def extract_claims(self, transcript: TranscriptionResult) -> List[AtomicClaim]:
        """
        Extracts verifiable claims from the transcription result.
        """
        if not transcript.has_speech or not transcript.full_text.strip():
            return []

        raw_text = transcript.full_text
        # Split by sentence terminators or strong conjunctions
        raw_sentences = re.split(r"(?<=[.!?])\s+|(?<=[,;])\s+(?=and\s+|but\s+|which\s+|where\s+)", raw_text)

        claims: List[AtomicClaim] = []
        claim_counter = 1

        for sentence in raw_sentences:
            cleaned = self._clean_sentence(sentence)
            if self._is_verifiable_claim(cleaned):
                # Estimate segment timestamp
                start_t = 0.0
                end_t = transcript.duration_sec
                for seg in transcript.segments:
                    if cleaned.lower() in seg.text.lower() or seg.text.lower() in cleaned.lower():
                        start_t = seg.start_sec
                        end_t = seg.end_sec
                        break

                claims.append(
                    AtomicClaim(
                        claim_id=f"claim_{claim_counter:03d}",
                        text=cleaned,
                        start_sec=round(start_t, 2),
                        end_sec=round(end_t, 2),
                        confidence=0.90
                    )
                )
                claim_counter += 1

                if len(claims) >= self.max_claims:
                    break

        # Fallback if no specific sentence passed strict filtering
        if not claims and len(raw_text.strip()) > 10:
            claims.append(
                AtomicClaim(
                    claim_id="claim_001",
                    text=raw_text.strip(),
                    start_sec=0.0,
                    end_sec=round(transcript.duration_sec, 2),
                    confidence=0.80
                )
            )

        return claims

    def _clean_sentence(self, sentence: str) -> str:
        s = sentence.strip()
        s = self.fillers.sub("", s)
        s = s.strip()
        if s and not s[-1] in ".!?":
            s += "."
        return s

    def _is_verifiable_claim(self, sentence: str) -> bool:
        """Heuristic check for factual assertiveness."""
        words = sentence.split()
        if len(words) < 4:
            return False
        # Ignore pure questions or greetings
        if sentence.endswith("?") or sentence.lower().startswith(("hello", "hi ", "thank you", "goodbye")):
            return False
        return True
