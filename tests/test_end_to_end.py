"""
End-to-End Pipeline Integration Tests
Generates synthetic media fixtures and verifies full multimodal pipeline execution.
"""

import os
import tempfile
import cv2
import numpy as np
import pytest

from src.pipeline import MultimodalVerificationPipeline


@pytest.fixture
def sample_video_path():
    """Generates a synthetic 1-second 30fps MP4 video fixture with a face-like circle."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 30.0, (320, 240))

    for i in range(30):
        frame = np.ones((240, 320, 3), dtype=np.uint8) * 180
        # Draw face oval
        cv2.ellipse(frame, (160, 120), (50, 70), 0, 0, 360, (220, 200, 180), -1)
        # Draw eyes
        cv2.circle(frame, (140, 105), 8, (50, 50, 50), -1)
        cv2.circle(frame, (180, 105), 8, (50, 50, 50), -1)
        # Draw mouth
        cv2.ellipse(frame, (160, 150), (20, 10), 0, 0, 180, (40, 40, 150), 2)
        out.write(frame)

    out.release()
    yield path

    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def test_pipeline_end_to_end_cheapfake(sample_video_path):
    pipeline = MultimodalVerificationPipeline()

    # Cheapfake test: pristine video paired with false viral claim
    false_claim = "Volodymyr Zelensky announced surrender to Russian forces in March 2022."
    report = pipeline.analyze_media(sample_video_path, override_claim_text=false_claim)

    assert report.s_sem >= 0.60
    assert report.semantic_fact_check["contradictions"] >= 1
    assert "Cheapfake" in report.verdict or "Misinformation" in report.verdict
    assert report.risk_level in ["HIGH", "CRITICAL"]

    report_dict = report.to_dict()
    assert "visual_forensics" in report_dict
    assert "semantic_fact_check" in report_dict
    assert "timeline" in report_dict["visual_forensics"]
