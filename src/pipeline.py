"""
Multimodal Deepfake Detection & Fact-Checking Pipeline Orchestrator
Integrates Ingestion, Visual ViT Forensics, Whisper ASR, RAG Evidence Retrieval,
NLI Verification, and Adaptive Confidence Gating into a unified pipeline.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import os
import yaml

from src.ingestion.video_extractor import VideoExtractor
from src.ingestion.audio_extractor import AudioExtractor
from src.visual_branch.visual_scorer import VisualScorer, VisualForensicResult
from src.semantic_branch.whisper_transcriber import WhisperTranscriber, TranscriptionResult
from src.semantic_branch.claim_extractor import ClaimExtractor
from src.semantic_branch.vector_store import VectorKnowledgeStore, DenseEmbeddingEncoder
from src.semantic_branch.evidence_retriever import EvidenceRetriever
from src.semantic_branch.nli_verifier import NLIVerifier, SemanticFactCheckResult
from src.fusion.adjudicator import ForensicAdjudicator, AdjudicationVerdict


@dataclass
class MultimodalVerificationReport:
    video_path: str
    verdict: str
    risk_level: str
    case_category: str
    s_final: float
    s_vis: float
    c_vis: float
    s_sem: float
    c_sem: float
    narrative_explanation: str
    recommended_action: str
    visual_forensics: Dict[str, Any]
    semantic_fact_check: Dict[str, Any]
    transcript: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultimodalVerificationPipeline:
    """
    End-to-End Multimodal Deepfake and Cheapfake Verification Engine.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)

        # Ingestion
        ingest_cfg = self.config.get("ingestion", {})
        self.video_extractor = VideoExtractor(
            sample_fps=ingest_cfg.get("sample_fps", 2.0),
            target_size=tuple(ingest_cfg.get("face_crop_size", [224, 224])),
            max_duration_sec=ingest_cfg.get("max_video_duration_sec", 120.0)
        )
        self.audio_extractor = AudioExtractor(
            target_sample_rate=ingest_cfg.get("audio_sample_rate", 16000)
        )

        # Visual Branch
        vis_cfg = self.config.get("visual_branch", {})
        self.visual_scorer = VisualScorer(
            deepfake_threshold=vis_cfg.get("deepfake_threshold", 0.50),
            device=self.config.get("system", {}).get("device", "auto")
        )

        # Semantic Branch
        sem_cfg = self.config.get("semantic_branch", {})
        self.whisper = WhisperTranscriber(
            model_name=sem_cfg.get("whisper_model", "openai/whisper-tiny"),
            device=self.config.get("system", {}).get("device", "auto")
        )
        self.claim_extractor = ClaimExtractor(
            max_claims=sem_cfg.get("max_atomic_claims", 5)
        )

        # Vector Knowledge Store & Retrieval
        corpus_path = sem_cfg.get("knowledge_corpus_path", "data/knowledge_base/fact_checks_corpus.json")
        encoder = DenseEmbeddingEncoder(
            model_name=sem_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.vector_store = VectorKnowledgeStore(corpus_path=corpus_path, encoder=encoder)
        self.retriever = EvidenceRetriever(
            vector_store=self.vector_store,
            top_k=sem_cfg.get("top_k_evidence", 3)
        )

        # NLI Verifier
        self.nli_verifier = NLIVerifier(
            model_name=sem_cfg.get("nli_model", "cross-encoder/nli-deberta-v3-small"),
            device=self.config.get("system", {}).get("device", "auto")
        )

        # Fusion & Adjudication
        self.adjudicator = ForensicAdjudicator()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "configs", "default_config.yaml"
        )
        target = config_path or default_path
        if os.path.exists(target):
            with open(target, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def analyze_media(
        self,
        video_path: str,
        override_claim_text: Optional[str] = None
    ) -> MultimodalVerificationReport:
        """
        Executes end-to-end multimodal verification on the input media file.
        """
        # 1. Ingestion: Video Frames
        frames = self.video_extractor.extract_frames(video_path)

        # 2. Ingestion: Audio Track
        audio = self.audio_extractor.extract_audio(video_path)

        # 3. Branch A: Visual Forgery Scoring
        vis_result: VisualForensicResult = self.visual_scorer.evaluate_video_frames(frames)

        # 4. Branch B: Whisper ASR Transcription
        transcription: TranscriptionResult = self.whisper.transcribe(
            audio, fallback_text=override_claim_text
        )

        # 5. Branch B: Claim Decomposition
        claims = self.claim_extractor.extract_claims(transcription)

        # 6. Branch B: Dense Evidence Retrieval
        evidence_lists = self.retriever.retrieve_batch(claims)

        # 7. Branch B: NLI Cross-Encoder Verification
        sem_result: SemanticFactCheckResult = self.nli_verifier.verify_claims(
            claims, evidence_lists
        )

        # 8. Stage 3: Adaptive Confidence Gating & Forensic Adjudication
        verdict: AdjudicationVerdict = self.adjudicator.adjudicate(
            s_vis=vis_result.s_vis,
            c_vis=vis_result.c_vis,
            s_sem=sem_result.s_sem,
            c_sem=sem_result.c_sem,
            peak_zone=vis_result.peak_anatomical_zone,
            peak_timestamp=vis_result.peak_frame_timestamp,
            num_contradictions=sem_result.num_contradictions
        )

        # 9. Format Unified Report
        report = MultimodalVerificationReport(
            video_path=video_path,
            verdict=verdict.primary_verdict,
            risk_level=verdict.risk_level,
            case_category=verdict.case_category,
            s_final=verdict.fusion_scores.s_final,
            s_vis=verdict.fusion_scores.s_vis,
            c_vis=verdict.fusion_scores.c_vis,
            s_sem=verdict.fusion_scores.s_sem,
            c_sem=verdict.fusion_scores.c_sem,
            narrative_explanation=verdict.narrative_explanation,
            recommended_action=verdict.recommended_action,
            visual_forensics={
                "s_vis": vis_result.s_vis,
                "c_vis": vis_result.c_vis,
                "is_manipulated": vis_result.is_manipulated,
                "num_frames": vis_result.num_frames_inspected,
                "num_faces": vis_result.num_faces_detected,
                "peak_zone": vis_result.peak_anatomical_zone,
                "peak_time_sec": vis_result.peak_frame_timestamp,
                "diagnostic": vis_result.diagnostic_summary,
                "timeline": vis_result.timeline
            },
            semantic_fact_check={
                "s_sem": sem_result.s_sem,
                "c_sem": sem_result.c_sem,
                "num_claims": sem_result.num_claims_analyzed,
                "contradictions": sem_result.num_contradictions,
                "diagnostic": sem_result.semantic_diagnostic,
                "claims": [
                    {
                        "claim_id": c.claim_id,
                        "claim_text": c.claim_text,
                        "timestamp": c.timestamp_range,
                        "best_evidence": c.best_evidence,
                        "source": c.evidence_source,
                        "contradiction_prob": c.contradiction_prob,
                        "entailment_prob": c.entailment_prob,
                        "verdict": c.ground_truth_verdict,
                        "is_contradicted": c.is_contradicted
                    }
                    for c in sem_result.claims_details
                ]
            },
            transcript=transcription.full_text
        )

        return report
