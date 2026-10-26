"""
Visual Forgery Scorer & Temporal Aggregation Module
Aggregates frame-level ViT predictions into video-level Visual Forgery Score S_vis
and confidence metric C_vis according to the technical dossier specifications.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import torch

from src.ingestion.video_extractor import FrameData, VideoExtractor
from .vit_detector import ViTArtifactInspector


@dataclass
class VisualForensicResult:
    s_vis: float                         # Visual Forgery Score in [0, 1]
    c_vis: float                         # Confidence in [0, 1]
    is_manipulated: bool
    num_frames_inspected: int
    num_faces_detected: int
    peak_frame_timestamp: float
    peak_anatomical_zone: str
    timeline: List[Dict[str, Any]]
    diagnostic_summary: str


class VisualScorer:
    """
    Computes calibrated visual forgery score S_vis and confidence C_vis
    across all sampled video frames.
    """

    def __init__(
        self,
        inspector: Optional[ViTArtifactInspector] = None,
        deepfake_threshold: float = 0.50,
        device: str = "auto"
    ):
        self.inspector = inspector or ViTArtifactInspector(device=device)
        self.deepfake_threshold = deepfake_threshold

    def evaluate_video_frames(self, frames: List[FrameData]) -> VisualForensicResult:
        """
        Processes a sequence of extracted frames, passes valid face crops to ViT,
        and aggregates temporal predictions.
        """
        valid_frames = [f for f in frames if f.face_crop is not None]

        if not valid_frames:
            return VisualForensicResult(
                s_vis=0.0,
                c_vis=0.1,
                is_manipulated=False,
                num_frames_inspected=len(frames),
                num_faces_detected=0,
                peak_frame_timestamp=0.0,
                peak_anatomical_zone="none",
                timeline=[],
                diagnostic_summary="No discernible facial regions found in the sampled video."
            )

        crops = [f.face_crop for f in valid_frames]
        tensor = VideoExtractor.to_torch_tensor(crops)

        batch_result = self.inspector.inspect_crops(tensor)
        per_frame = batch_result["per_frame_results"]

        timeline = []
        scores = []
        peak_score = -1.0
        peak_idx = 0

        for idx, (frame_info, vit_res) in enumerate(zip(valid_frames, per_frame)):
            fake_p = vit_res["fake_probability"]
            scores.append(fake_p)
            if fake_p > peak_score:
                peak_score = fake_p
                peak_idx = idx

            timeline.append({
                "frame_index": frame_info.frame_index,
                "timestamp_sec": frame_info.timestamp_sec,
                "fake_prob": round(fake_p, 4),
                "is_fake": vit_res["is_fake"],
                "bbox": frame_info.bbox,
                "peak_zone": vit_res["peak_anomaly_coordinate"]["anatomical_zone"]
            })

        scores_arr = np.array(scores)

        # Robust aggregation: 70% weighted top-k percentiles + 30% mean
        # This prevents a brief 1-frame transient artifact from dominating while ensuring
        # localized surgical face swaps (which appear on high-motion frames) are detected.
        top_k_scores = np.sort(scores_arr)[-max(1, len(scores_arr) // 3):]
        s_vis = float(0.70 * np.mean(top_k_scores) + 0.30 * np.mean(scores_arr))
        s_vis = float(np.clip(s_vis, 0.0, 1.0))

        # Confidence C_vis is driven by sample density and prediction sharpness:
        # C_vis is high when predictions are decisively near 0.0 or 1.0 across multiple frames
        sample_factor = min(1.0, len(valid_frames) / 5.0)
        certainty = float(np.mean(np.abs(scores_arr - 0.5) * 2.0))
        c_vis = float(np.clip(0.4 * sample_factor + 0.6 * certainty, 0.1, 0.99))

        is_manipulated = s_vis >= self.deepfake_threshold
        peak_zone = per_frame[peak_idx]["peak_anomaly_coordinate"]["anatomical_zone"]
        peak_time = valid_frames[peak_idx].timestamp_sec

        if is_manipulated:
            diagnostic = (
                f"Visual forgery detected (S_vis: {s_vis:.2f}, Conf: {c_vis:.2f}). "
                f"Peak synthesis anomalies localized around {peak_zone.replace('_', ' ')} "
                f"at t={peak_time:.1f}s."
            )
        else:
            diagnostic = (
                f"Visual frames appear authentic (S_vis: {s_vis:.2f}, Conf: {c_vis:.2f}). "
                f"Facial texture and boundary gradients conform to genuine optical distributions."
            )

        return VisualForensicResult(
            s_vis=round(s_vis, 4),
            c_vis=round(c_vis, 4),
            is_manipulated=is_manipulated,
            num_frames_inspected=len(frames),
            num_faces_detected=len(valid_frames),
            peak_frame_timestamp=peak_time,
            peak_anatomical_zone=peak_zone,
            timeline=timeline,
            diagnostic_summary=diagnostic
        )
