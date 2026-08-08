"""
Head Pose Estimation Module.
Uses cv2.solvePnP with a 3D canonical facial model to compute Head Yaw, Pitch, and Roll angles.
"""

import numpy as np
import cv2


class HeadPoseEstimator:
    """Estimates head pose (Yaw, Pitch, Roll in degrees) using 3D-to-2D facial landmark matching."""

    # 3D Canonical Facial Model Points (x, y, z)
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),             # Nose Tip (landmark 1)
        (0.0, -330.0, -65.0),        # Chin (landmark 152)
        (-225.0, 170.0, -135.0),     # Left Eye Outer Corner (landmark 33)
        (225.0, 170.0, -135.0),      # Right Eye Outer Corner (landmark 263)
        (-150.0, -150.0, -125.0),    # Left Mouth Corner (landmark 61)
        (150.0, -150.0, -125.0)      # Right Mouth Corner (landmark 291)
    ], dtype=np.float64)

    # Corresponding MediaPipe landmark indices
    LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

    def estimate_pose(self, landmarks_2d: np.ndarray, frame_size: tuple[int, int]) -> dict:
        """
        Computes head pose Euler angles (yaw, pitch, roll in degrees).
        """
        w, h = frame_size

        # Extract 2D coordinates for key pose landmarks
        image_points_2d = landmarks_2d[self.LANDMARK_INDICES].astype(np.float64)

        # Approximate Camera Intrinsic Matrix
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve Perspective-n-Point
        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "rvec": None, "tvec": None}

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix into Euler angles
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
            yaw = np.arctan2(-rmat[2, 0], sy)
            roll = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
            yaw = np.arctan2(-rmat[2, 0], sy)
            roll = 0.0

        # Convert radians to degrees
        yaw_deg = float(np.degrees(yaw))
        pitch_deg = float(np.degrees(pitch))
        roll_deg = float(np.degrees(roll))

        return {
            "yaw": yaw_deg,
            "pitch": pitch_deg,
            "roll": roll_deg,
            "rvec": rvec,
            "tvec": tvec,
            "camera_matrix": camera_matrix,
            "nose_tip_2d": image_points_2d[0]
        }

    def draw_pose_axes(self, frame: np.ndarray, pose_data: dict, length: float = 50.0) -> None:
        """Draws 3D coordinate orientation axes (X=Red, Y=Green, Z=Blue) at the nose tip."""
        rvec = pose_data.get("rvec")
        tvec = pose_data.get("tvec")
        camera_matrix = pose_data.get("camera_matrix")

        if rvec is None or tvec is None or camera_matrix is None:
            return

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 3D axis endpoints
        axis_3d = np.array([
            (length, 0.0, 0.0),    # X-axis (Pitch / Red)
            (0.0, length, 0.0),    # Y-axis (Yaw / Green)
            (0.0, 0.0, length)     # Z-axis (Roll / Blue)
        ], dtype=np.float64)

        # Project 3D axis points onto 2D image plane
        axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)

        nose_pt = tuple(pose_data["nose_tip_2d"].astype(int))
        p_x = tuple(axis_2d[0].ravel().astype(int))
        p_y = tuple(axis_2d[1].ravel().astype(int))
        p_z = tuple(axis_2d[2].ravel().astype(int))

        # Render 3D coordinate frame lines
        cv2.line(frame, nose_pt, p_x, (0, 0, 255), 2)  # X-axis: Red
        cv2.line(frame, nose_pt, p_y, (0, 255, 0), 2)  # Y-axis: Green
        cv2.line(frame, nose_pt, p_z, (255, 0, 0), 2)  # Z-axis: Blue
