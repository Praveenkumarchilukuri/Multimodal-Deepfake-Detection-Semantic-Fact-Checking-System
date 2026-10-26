"""
Unit Tests for Visual Branch (ViT Detector, Artifact Inspector, Visual Scorer)
"""

import numpy as np
import pytest
import torch

from src.visual_branch.vit_detector import (
    ViTDeepfakeClassifier,
    ViTArtifactInspector,
    PatchEmbedding
)
from src.visual_branch.visual_scorer import VisualScorer
from src.ingestion.video_extractor import FrameData


def test_patch_embedding_dimensions():
    embed = PatchEmbedding(in_channels=3, patch_size=16, embed_dim=128)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = embed(dummy_input)

    # 224 / 16 = 14 -> 14 * 14 = 196 patches
    assert out.shape == (2, 196, 128)


def test_vit_classifier_forward():
    model = ViTDeepfakeClassifier(
        img_size=224, patch_size=16, embed_dim=128, depth=2, num_heads=4, num_classes=2
    )
    dummy_input = torch.randn(2, 3, 224, 224)
    logits, patches = model(dummy_input)

    assert logits.shape == (2, 2)
    assert patches.shape == (2, 196, 128)


def test_vit_inspector_spatial_heatmap():
    model = ViTDeepfakeClassifier(
        img_size=224, patch_size=16, embed_dim=128, depth=2, num_heads=4, num_classes=2
    )
    inspector = ViTArtifactInspector(model=model, device="cpu")

    crops = torch.randn(3, 3, 224, 224)
    result = inspector.inspect_crops(crops)

    assert result["batch_size"] == 3
    assert len(result["per_frame_results"]) == 3

    # Check 14x14 heatmap
    heatmap = np.array(result["per_frame_results"][0]["anomaly_heatmap_14x14"])
    assert heatmap.shape == (14, 14)
    assert 0.0 <= np.min(heatmap) <= np.max(heatmap) <= 1.0


def test_visual_scorer_aggregation():
    model = ViTDeepfakeClassifier(
        img_size=224, patch_size=16, embed_dim=128, depth=2, num_heads=4, num_classes=2
    )
    inspector = ViTArtifactInspector(model=model, device="cpu")
    scorer = VisualScorer(inspector=inspector, deepfake_threshold=0.50)

    # Mock frames
    frames = [
        FrameData(
            frame_index=i,
            timestamp_sec=i * 0.5,
            raw_frame=np.zeros((224, 224, 3), dtype=np.uint8),
            face_crop=np.ones((224, 224, 3), dtype=np.uint8) * 100,
            bbox=(10, 10, 100, 100),
            has_face=True
        )
        for i in range(4)
    ]

    res = scorer.evaluate_video_frames(frames)
    assert 0.0 <= res.s_vis <= 1.0
    assert 0.0 <= res.c_vis <= 1.0
    assert res.num_frames_inspected == 4
    assert res.num_faces_detected == 4
