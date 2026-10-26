"""
Adaptive Confidence Gating Module
Implements non-linear cross-modal fusion:
S_final = σ( α(C_vis)·S_vis + β(C_sem)·S_sem + γ·(S_vis ⊙ S_sem) )
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class FusionScores:
    s_vis: float
    c_vis: float
    s_sem: float
    c_sem: float
    alpha_effective: float
    beta_effective: float
    cross_modal_term: float
    raw_logit: float
    s_final: float


class AdaptiveConfidenceGating:
    """
    Computes unified multimodal risk score using confidence-gated weights
    and multiplicative cross-modal interaction terms.
    """

    def __init__(
        self,
        alpha_base: float = 0.50,
        beta_base: float = 0.40,
        gamma_cross: float = 0.20,
        temperature: float = 1.0
    ):
        self.alpha_base = alpha_base
        self.beta_base = beta_base
        self.gamma_cross = gamma_cross
        self.temperature = temperature

    def fuse(
        self,
        s_vis: float,
        c_vis: float,
        s_sem: float,
        c_sem: float
    ) -> FusionScores:
        """
        Fuses visual and semantic signals into S_final.
        """
        # Dynamic weighting scaling with branch confidence:
        # High confidence amplifies the branch influence
        alpha_eff = self.alpha_base * (1.0 + 0.5 * c_vis)
        beta_eff = self.beta_base * (1.0 + 0.5 * c_sem)

        # Cross-modal interaction term (Hadamard product)
        cross_term = self.gamma_cross * (s_vis * s_sem)

        # Scaled logit representation
        raw_linear = (alpha_eff * s_vis + beta_eff * s_sem + cross_term)
        # Shift to center around 0.5 prior
        logit = (raw_linear - (alpha_eff * 0.5 + beta_eff * 0.5)) / max(0.1, self.temperature)

        # Sigmoid activation: σ(z) = 1 / (1 + exp(-z))
        s_final = float(1.0 / (1.0 + np.exp(-logit * 3.0)))
        s_final = float(np.clip(s_final, 0.0, 1.0))

        return FusionScores(
            s_vis=round(s_vis, 4),
            c_vis=round(c_vis, 4),
            s_sem=round(s_sem, 4),
            c_sem=round(c_sem, 4),
            alpha_effective=round(alpha_eff, 4),
            beta_effective=round(beta_eff, 4),
            cross_modal_term=round(cross_term, 4),
            raw_logit=round(logit, 4),
            s_final=round(s_final, 4)
        )
