"""
Unit Tests for Adaptive Confidence Gating & Forensic Adjudication
"""

import pytest
from src.fusion.adaptive_gating import AdaptiveConfidenceGating
from src.fusion.adjudicator import ForensicAdjudicator


def test_adaptive_gating_math():
    gating = AdaptiveConfidenceGating(alpha_base=0.5, beta_base=0.4, gamma_cross=0.2)
    scores = gating.fuse(s_vis=0.8, c_vis=0.9, s_sem=0.2, c_sem=0.8)

    assert 0.0 <= scores.s_final <= 1.0
    assert scores.alpha_effective > gating.alpha_base
    assert scores.beta_effective > gating.beta_base
    assert scores.cross_modal_term > 0.0


def test_adjudicator_case_1_visual_deepfake():
    adjudicator = ForensicAdjudicator()
    # High visual forgery (0.85), Low semantic inconsistency (0.15)
    verdict = adjudicator.adjudicate(
        s_vis=0.85, c_vis=0.90, s_sem=0.15, c_sem=0.85, peak_zone="jawline_and_chin_boundary"
    )

    assert verdict.case_category == "CASE_1_VISUAL_DEEPFAKE"
    assert "Visual Deepfake" in verdict.primary_verdict
    assert verdict.risk_level == "HIGH"


def test_adjudicator_case_2_cheapfake():
    adjudicator = ForensicAdjudicator()
    # Clean visual pixels (0.10), High semantic inconsistency (0.88)
    verdict = adjudicator.adjudicate(
        s_vis=0.10, c_vis=0.95, s_sem=0.88, c_sem=0.90, num_contradictions=2
    )

    assert verdict.case_category == "CASE_2_CHEAPFAKE_MISINFORMATION"
    assert "Cheapfake" in verdict.primary_verdict
    assert verdict.risk_level == "HIGH"


def test_adjudicator_case_3_composite():
    adjudicator = ForensicAdjudicator()
    # High visual forgery (0.85) AND High semantic inconsistency (0.90)
    verdict = adjudicator.adjudicate(
        s_vis=0.85, c_vis=0.90, s_sem=0.90, c_sem=0.90, num_contradictions=2
    )

    assert verdict.case_category == "CASE_3_COMPOSITE_FORGERY"
    assert verdict.risk_level == "CRITICAL"


def test_adjudicator_case_4_authentic():
    adjudicator = ForensicAdjudicator()
    # Clean visual pixels (0.15), Low semantic inconsistency (0.10)
    verdict = adjudicator.adjudicate(
        s_vis=0.15, c_vis=0.95, s_sem=0.10, c_sem=0.90
    )

    assert verdict.case_category == "CASE_4_AUTHENTIC_MEDIA"
    assert verdict.risk_level == "LOW"
