"""
Unit Tests for Ingestion Module (Video & Audio Extractors)
"""

import os
import numpy as np
import pytest
import torch

from src.ingestion.video_extractor import VideoExtractor, FrameData
from src.ingestion.audio_extractor import AudioExtractor, AudioData


def test_video_extractor_center_crop():
    extractor = VideoExtractor(sample_fps=2.0, target_size=(224, 224))
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    crop, bbox, has_face = extractor._detect_and_crop_face(dummy_frame)
    assert crop.shape == (224, 224, 3)
    assert crop.dtype == np.uint8


def test_to_torch_tensor_normalization():
    crops = [np.ones((224, 224, 3), dtype=np.uint8) * 128 for _ in range(3)]
    tensor = VideoExtractor.to_torch_tensor(crops)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_audio_extractor_empty_fallback():
    extractor = AudioExtractor(target_sample_rate=16000)
    # Non-existent or dummy file test
    with pytest.raises(FileNotFoundError):
        extractor.extract_audio("non_existent_file.mp4")


def test_audio_resampling():
    extractor = AudioExtractor(target_sample_rate=16000)
    waveform_44k = np.sin(np.linspace(0, 100, 44100)).astype(np.float32)
    resampled = extractor._resample_to_16k(waveform_44k, 44100)

    assert len(resampled) == 16000
    assert resampled.dtype == np.float32
