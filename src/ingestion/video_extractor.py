"""
Video Ingestion & Frame Preprocessing Module
Extracts video frames at specified sample rates, detects face regions,
and aligns crops into standard 224x224 RGB tensors for Vision Transformer analysis.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
import cv2
import numpy as np
import torch


@dataclass
class FrameData:
    frame_index: int
    timestamp_sec: float
    raw_frame: np.ndarray  # HxWxC BGR
    face_crop: Optional[np.ndarray]  # 224x224x3 RGB
    bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    has_face: bool


class VideoExtractor:
    """
    Handles streaming and temporal frame sampling from video files.
    Isolates facial regions of interest and standardizes geometry for ViT.
    """

    def __init__(
        self,
        sample_fps: float = 2.0,
        target_size: Tuple[int, int] = (224, 224),
        max_duration_sec: float = 120.0
    ):
        self.sample_fps = sample_fps
        self.target_size = target_size
        self.max_duration_sec = max_duration_sec

        # Built-in OpenCV Haar Cascade for robust zero-dependency face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.face_cascade = None

    def extract_frames(self, video_path: str) -> List[FrameData]:
        """
        Samples frames at self.sample_fps, extracts face crops, and returns metadata.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video stream: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        duration_sec = total_frames / fps
        effective_duration = min(duration_sec, self.max_duration_sec)

        # Calculate frame step for desired sample_fps
        frame_step = max(1, int(round(fps / self.sample_fps)))

        extracted: List[FrameData] = []
        current_frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_sec = current_frame_idx / fps
            if timestamp_sec > self.max_duration_sec:
                break

            if current_frame_idx % frame_step == 0:
                face_crop, bbox, has_face = self._detect_and_crop_face(frame)
                extracted.append(
                    FrameData(
                        frame_index=current_frame_idx,
                        timestamp_sec=round(timestamp_sec, 3),
                        raw_frame=frame,
                        face_crop=face_crop,
                        bbox=bbox,
                        has_face=has_face
                    )
                )

            current_frame_idx += 1

        cap.release()
        return extracted

    def _detect_and_crop_face(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], bool]:
        """
        Detects primary face in frame. If no face detected, extracts central crop.
        Returns (224x224 RGB image, bbox, has_detected_face).
        """
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bbox = None
        has_face = False

        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
            )
            if len(faces) > 0:
                # Select the largest face by area
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, fw, fh = faces[0]

                # Apply slight margin expansion (15%) for boundary artifact detection
                margin_x = int(0.15 * fw)
                margin_y = int(0.15 * fh)
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(w, x + fw + margin_x)
                y2 = min(h, y + fh + margin_y)

                face_region = frame[y1:y2, x1:x2]
                bbox = (x, y, fw, fh)
                has_face = True
            else:
                face_region = self._center_crop(frame)
        else:
            face_region = self._center_crop(frame)

        # Standardize to RGB and resize to target_size (224x224)
        rgb_crop = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_crop, self.target_size, interpolation=cv2.INTER_AREA)

        return resized, bbox, has_face

    def _center_crop(self, frame: np.ndarray) -> np.ndarray:
        """Fallback square center crop if no face detected."""
        h, w, _ = frame.shape
        min_dim = min(h, w)
        start_y = (h - min_dim) // 2
        start_x = (w - min_dim) // 2
        return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

    @staticmethod
    def to_torch_tensor(crops: List[np.ndarray]) -> torch.Tensor:
        """
        Converts a list of (224, 224, 3) RGB uint8 numpy images
        into a normalized PyTorch tensor [B, 3, 224, 224] using standard ImageNet stats.
        """
        if not crops:
            return torch.empty(0, 3, 224, 224, dtype=torch.float32)

        # Convert uint8 [0, 255] to float32 [0.0, 1.0]
        batch_np = np.stack(crops, axis=0).astype(np.float32) / 255.0
        # Transpose from [B, H, W, C] to [B, C, H, W]
        batch_np = np.transpose(batch_np, (0, 3, 1, 2))
        tensor = torch.from_numpy(batch_np)

        # Normalize with ImageNet mean and standard deviation
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        normalized = (tensor - mean) / std
        return normalized
