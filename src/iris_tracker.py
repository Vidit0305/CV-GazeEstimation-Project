"""
Iris Tracker Module.
Calculates scale-invariant, normalized relative iris coordinates (iris_x, iris_y)
inside left and right eye boundaries using geometric vector projections.
"""

import numpy as np
import cv2
from src.eye_tracker import EyeTracker


class IrisTracker:
    """Tracks left and right iris centers and computes normalized eye-relative positions."""

    # Landmark indices for MediaPipe Iris Tracking
    LEFT_IRIS_CENTER = 468
    LEFT_IRIS_CONTOUR = [469, 470, 471, 472]

    RIGHT_IRIS_CENTER = 473
    RIGHT_IRIS_CONTOUR = [474, 475, 476, 477]

    @staticmethod
    def _project_relative_pos(
        iris_pt: np.ndarray,
        p_start: np.ndarray,
        p_end: np.ndarray
    ) -> float:
        """
        Projects iris_pt onto the line segment (p_start -> p_end).
        Returns a normalized scalar value between 0.0 and 1.0.
        """
        v_line = p_end[:2] - p_start[:2]
        line_len = np.linalg.norm(v_line)

        if line_len < 1e-6:
            return 0.5

        unit_line = v_line / line_len
        v_iris = iris_pt[:2] - p_start[:2]
        proj = float(np.dot(v_iris, unit_line))

        return float(np.clip(proj / line_len, 0.0, 1.0))

    def process_irises(self, landmarks_2d: np.ndarray) -> dict:
        """
        Calculates normalized iris_x and iris_y for left and right eyes.
        Returns dictionary containing iris centers, contours, and normalized coordinates.
        """
        left_iris_center = landmarks_2d[self.LEFT_IRIS_CENTER]
        right_iris_center = landmarks_2d[self.RIGHT_IRIS_CENTER]

        # Left Eye Boundaries
        l_outer = landmarks_2d[EyeTracker.LEFT_EYE_CORNER_OUTER]
        l_inner = landmarks_2d[EyeTracker.LEFT_EYE_CORNER_INNER]
        l_top = landmarks_2d[EyeTracker.LEFT_EYE_TOP]
        l_bottom = landmarks_2d[EyeTracker.LEFT_EYE_BOTTOM]

        # Right Eye Boundaries
        r_inner = landmarks_2d[EyeTracker.RIGHT_EYE_CORNER_INNER]
        r_outer = landmarks_2d[EyeTracker.RIGHT_EYE_CORNER_OUTER]
        r_top = landmarks_2d[EyeTracker.RIGHT_EYE_TOP]
        r_bottom = landmarks_2d[EyeTracker.RIGHT_EYE_BOTTOM]

        # Calculate Normalized Relative Coordinates
        left_iris_x = self._project_relative_pos(left_iris_center, l_outer, l_inner)
        left_iris_y = self._project_relative_pos(left_iris_center, l_top, l_bottom)

        right_iris_x = self._project_relative_pos(right_iris_center, r_inner, r_outer)
        right_iris_y = self._project_relative_pos(right_iris_center, r_top, r_bottom)

        left_contour_pts = landmarks_2d[self.LEFT_IRIS_CONTOUR].astype(np.int32)
        right_contour_pts = landmarks_2d[self.RIGHT_IRIS_CONTOUR].astype(np.int32)

        return {
            "left_iris_center": left_iris_center,
            "right_iris_center": right_iris_center,
            "left_iris_x": left_iris_x,
            "left_iris_y": left_iris_y,
            "right_iris_x": right_iris_x,
            "right_iris_y": right_iris_y,
            "left_contour": left_contour_pts,
            "right_contour": right_contour_pts
        }

    def draw_irises(
        self,
        frame: np.ndarray,
        iris_data: dict,
        color: tuple[int, int, int] = (0, 255, 255),
        center_color: tuple[int, int, int] = (0, 0, 255)
    ) -> None:
        """Draws iris center dots and iris boundary rings on webcam frame."""
        l_center = tuple(iris_data["left_iris_center"][:2].astype(int))
        r_center = tuple(iris_data["right_iris_center"][:2].astype(int))

        # Draw boundary rings
        cv2.polylines(frame, [iris_data["left_contour"]], isClosed=True, color=color, thickness=1)
        cv2.polylines(frame, [iris_data["right_contour"]], isClosed=True, color=color, thickness=1)

        # Draw iris centers
        cv2.circle(frame, l_center, 2, center_color, -1)
        cv2.circle(frame, r_center, 2, center_color, -1)
