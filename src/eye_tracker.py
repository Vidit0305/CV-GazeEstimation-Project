"""
Eye Tracker Module.
Extracts left and right eye landmark contours, calculates Eye Aspect Ratio (EAR),
and evaluates tracking confidence based on eye opening metrics.
"""

import numpy as np
import cv2


class EyeTracker:
    """Manages eye landmark extraction, aspect ratio calculation, and eye quality diagnostics."""

    # Key landmark indices in MediaPipe Face Mesh
    LEFT_EYE_CORNER_OUTER = 33
    LEFT_EYE_CORNER_INNER = 133
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]

    RIGHT_EYE_CORNER_INNER = 362
    RIGHT_EYE_CORNER_OUTER = 263
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

    def __init__(self, ear_blink_threshold: float = 0.12):
        self.ear_blink_threshold = ear_blink_threshold

    @staticmethod
    def _euclidean_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
        """Calculates 2D Euclidean distance between two points."""
        return float(np.linalg.norm(pt1[:2] - pt2[:2]))

    def calculate_ear(self, landmarks_2d: np.ndarray, is_left: bool = True) -> float:
        """
        Calculates Eye Aspect Ratio (EAR) for left or right eye.
        EAR = (Vertical Distance) / (Horizontal Distance)
        """
        if is_left:
            top = landmarks_2d[self.LEFT_EYE_TOP]
            bottom = landmarks_2d[self.LEFT_EYE_BOTTOM]
            outer = landmarks_2d[self.LEFT_EYE_CORNER_OUTER]
            inner = landmarks_2d[self.LEFT_EYE_CORNER_INNER]
        else:
            top = landmarks_2d[self.RIGHT_EYE_TOP]
            bottom = landmarks_2d[self.RIGHT_EYE_BOTTOM]
            outer = landmarks_2d[self.RIGHT_EYE_CORNER_OUTER]
            inner = landmarks_2d[self.RIGHT_EYE_CORNER_INNER]

        v_dist = self._euclidean_distance(top, bottom)
        h_dist = self._euclidean_distance(outer, inner)

        if h_dist < 1e-6:
            return 0.0

        return v_dist / h_dist

    def process_eyes(self, landmarks_2d: np.ndarray) -> dict:
        """
        Extracts eye landmarks, EAR values, and blink status.
        """
        left_ear = self.calculate_ear(landmarks_2d, is_left=True)
        right_ear = self.calculate_ear(landmarks_2d, is_left=False)
        avg_ear = (left_ear + right_ear) / 2.0

        left_closed = left_ear < self.ear_blink_threshold
        right_closed = right_ear < self.ear_blink_threshold

        left_contour_pts = landmarks_2d[self.LEFT_EYE_CONTOUR].astype(np.int32)
        right_contour_pts = landmarks_2d[self.RIGHT_EYE_CONTOUR].astype(np.int32)

        return {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "avg_ear": avg_ear,
            "left_closed": left_closed,
            "right_closed": right_closed,
            "left_contour": left_contour_pts,
            "right_contour": right_contour_pts,
            "eyes_open": not (left_closed or right_closed)
        }

    def draw_eye_contours(
        self,
        frame: np.ndarray,
        eye_data: dict,
        color: tuple[int, int, int] = (0, 255, 200),
        thickness: int = 1
    ) -> None:
        """Draws debugging polygon outlines around left and right eyes."""
        cv2.polylines(frame, [eye_data["left_contour"]], isClosed=True, color=color, thickness=thickness)
        cv2.polylines(frame, [eye_data["right_contour"]], isClosed=True, color=color, thickness=thickness)
