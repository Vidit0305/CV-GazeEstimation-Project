"""
Face Tracker Module using MediaPipe FaceLandmarker Tasks API.
Extracts 478 3D facial landmarks including iris centers and contours.
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
import config


MODEL_PATH = os.path.join(config.MODELS_DIR, "face_landmarker.task")


def ensure_model_asset() -> str:
    """Ensures the face_landmarker.task model bundle is available locally."""
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Downloading MediaPipe FaceLandmarker model bundle to {MODEL_PATH}...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("[SUCCESS] Model bundle downloaded successfully.")
    return MODEL_PATH


class FaceTracker:
    """Detects 478 3D facial landmarks using MediaPipe Tasks API."""

    def __init__(
        self,
        max_num_faces: int = config.MAX_NUM_FACES,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE
    ):
        model_path = ensure_model_asset()

        base_options = BaseOptions(model_asset_path=model_path)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray) -> dict | None:
        """
        Processes a BGR image frame and extracts 2D/3D landmark coordinates.
        Returns a dictionary containing:
        - 'landmarks_2d': np.ndarray of shape (478, 2) in image pixel coordinates
        - 'landmarks_norm': np.ndarray of shape (478, 3) normalized (x, y, z)
        - 'frame_size': (width, height)
        Returns None if no face is detected.
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        results = self.landmarker.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return None

        landmarks = results.face_landmarks[0]

        norm_coords = np.array([
            [lm.x, lm.y, lm.z] for lm in landmarks
        ], dtype=np.float32)

        pixel_coords = np.array([
            [lm.x * w, lm.y * h] for lm in landmarks
        ], dtype=np.float32)

        return {
            "landmarks_2d": pixel_coords,
            "landmarks_norm": norm_coords,
            "frame_size": (w, h),
            "raw_landmarks": landmarks
        }

    def close(self) -> None:
        """Frees MediaPipe C++ runtime resources."""
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()
