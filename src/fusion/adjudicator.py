"""
Forensic Adjudication & Decision Matrix Module
Categorizes multimodal verification results into the 4 forensic cases:
- Case 1: Visual Deepfake Alert (Manipulated pixels + authentic claim)
- Case 2: Cheapfake / Misinformation Alert (Authentic pixels + false claim)
- Case 3: Composite Deepfake (Both pixels and narrative manipulated)
- Case 4: Verified Authentic Media
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from .adaptive_gating import AdaptiveConfidenceGating, FusionScores


@dataclass
class AdjudicationVerdict:
    case_category: str
    primary_verdict: str
    risk_level: str  # "CRITICAL", "HIGH", "MODERATE", "LOW"
    narrative_explanation: str
    recommended_action: str
    fusion_scores: FusionScores
    audit_metadata: Dict[str, Any]


class ForensicAdjudicator:
    """
    Applies non-linear case logic to determine final forensic verdict
    and generates auditable explanations.
    """

    def __init__(self, gating: Optional[AdaptiveConfidenceGating] = None):
        self.gating = gating or AdaptiveConfidenceGating()

    def adjudicate(
        self,
        s_vis: float,
        c_vis: float,
        s_sem: float,
        c_sem: float,
        peak_zone: str = "facial_center",
        peak_timestamp: float = 0.0,
        num_contradictions: int = 0
    ) -> AdjudicationVerdict:
        """
        Executes decision matrix over visual and semantic forensic scores.
        """
        fusion = self.gating.fuse(s_vis, c_vis, s_sem, c_sem)
        s_f = fusion.s_final

        # Case 3: Composite Deception (Both pixels and speech false)
        if s_vis >= 0.55 and s_sem >= 0.60:
            case_cat = "CASE_3_COMPOSITE_FORGERY"
            verdict = "Composite Deepfake & Misinformation Alert"
            risk = "CRITICAL"
            narrative = (
                f"Multi-layered deception detected. Visual synthesis artifacts identified "
                f"around {peak_zone.replace('_', ' ')} (S_vis: {s_vis:.2f}) coupled with "
                f"{num_contradictions} contradicted factual statements in the audio narrative (S_sem: {s_sem:.2f}). "
                f"Overall composite risk S_final: {s_f:.2f}."
            )
            action = "Block media immediately; issue critical multimodal synthetic deception advisory."

        # Case 1: Visual Deepfake Alert (Manipulated face + authentic/neutral audio)
        elif s_vis >= 0.50 and s_sem < 0.50:
            case_cat = "CASE_1_VISUAL_DEEPFAKE"
            verdict = "Visual Deepfake Alert"
            risk = "HIGH"
            narrative = (
                f"Pixel-level facial synthesis detected (S_vis: {s_vis:.2f}, Conf: {c_vis:.2f}). "
                f"Primary boundary and texture anomalies localized at {peak_zone.replace('_', ' ')} "
                f"(peak at t={peak_timestamp:.1f}s). The spoken speech content appears factual or neutral "
                f"(S_sem: {s_sem:.2f}), confirming identity-swap or visual face reenactment."
            )
            action = "Flag video for synthetic face manipulation (face-swap / reenactment); visual integrity compromised."

        # Case 2: Cheapfake / Misinformation Alert (Authentic pixels + false claim)
        elif s_vis < 0.50 and s_sem >= 0.60:
            case_cat = "CASE_2_CHEAPFAKE_MISINFORMATION"
            verdict = "Misinformation Alert / Cheapfake"
            risk = "HIGH"
            narrative = (
                f"Contextual deception / Cheapfake detected. Facial pixels appear genuine and unmanipulated "
                f"(S_vis: {s_vis:.2f}), but spoken statements strongly contradict verified ground truth "
                f"(S_sem: {s_sem:.2f}, Conf: {c_sem:.2f}, {num_contradictions} false assertions). "
                f"Genuine archival footage has been recontextualized or dubbed with false claims."
            )
            action = "Attach contextual fact-check label and debunking evidence; video pixels are authentic but narrative is false."

        # Case 4: Verified Authentic Media
        elif s_vis < 0.50 and s_sem < 0.50:
            case_cat = "CASE_4_AUTHENTIC_MEDIA"
            verdict = "Verified Authentic Media"
            risk = "LOW"
            narrative = (
                f"Media satisfies authenticity thresholds (S_final: {s_f:.2f}). "
                f"No synthetic visual artifacts detected (S_vis: {s_vis:.2f}) and spoken claims align "
                f"with verified knowledge archives (S_sem: {s_sem:.2f})."
            )
            action = "Mark as authentic and compliant with integrity standards."

        # Ambiguous / Boundary Case
        else:
            case_cat = "CASE_AMBIGUOUS_BORDERLINE"
            verdict = "Borderline / Elevated Scrutiny Required"
            risk = "MODERATE"
            narrative = (
                f"Inconclusive multimodal signals (S_final: {s_f:.2f}, S_vis: {s_vis:.2f}, S_sem: {s_sem:.2f}). "
                f"Requires secondary human review or additional source corroboration."
            )
            action = "Route to human forensic analyst for secondary review."

        audit_meta = {
            "peak_anatomical_zone": peak_zone,
            "peak_timestamp_sec": peak_timestamp,
            "contradiction_count": num_contradictions,
            "decision_confidence": round(float(0.5 * (c_vis + c_sem)), 4)
        }

        return AdjudicationVerdict(
            case_category=case_cat,
            primary_verdict=verdict,
            risk_level=risk,
            narrative_explanation=narrative,
            recommended_action=action,
            fusion_scores=fusion,
            audit_metadata=audit_meta
        )
