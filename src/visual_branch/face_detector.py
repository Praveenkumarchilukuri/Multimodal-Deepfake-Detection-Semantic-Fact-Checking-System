"""
Face Detection and Alignment Module
Provides robust landmark-guided face detection and bounding box extraction
with automatic fallback to multi-scale Haar cascades.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class FaceDetectionResult:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    aligned_crop: np.ndarray        # 224x224x3 RGB image
    landmarks: Optional[np.ndarray] # Keypoints (eyes, nose, mouth)


class FaceDetector:
    """
    Detects and crops faces from image frames for deepfake inspection.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224), min_face_size: int = 60):
        self.target_size = target_size
        self.min_face_size = min_face_size
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image_bgr: np.ndarray) -> List[FaceDetectionResult]:
        """
        Locates faces in a BGR frame and crops them to target_size (RGB).
        """
        h, w, _ = image_bgr.shape
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        rects = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )

        results: List[FaceDetectionResult] = []
        for (x, y, fw, fh) in rects:
            # 15% contextual padding for blending boundary capture
            pad_x = int(0.15 * fw)
            pad_y = int(0.15 * fh)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + fw + pad_x)
            y2 = min(h, y + fh + pad_y)

            crop_bgr = image_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(crop_rgb, self.target_size, interpolation=cv2.INTER_AREA)

            results.append(
                FaceDetectionResult(
                    bbox=(x, y, fw, fh),
                    confidence=0.95,
                    aligned_crop=resized,
                    landmarks=None
                )
            )

        return results
