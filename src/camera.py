"""
Webcam Stream Capture Module.
Handles camera initialization, HD resolution capture, horizontal mirroring, and error handling.
"""

import cv2
import numpy as np
import config


class WebcamStream:
    """Wrapper around OpenCV VideoCapture for webcam video streaming."""

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        flip_horizontal: bool = config.FLIP_HORIZONTAL
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.flip_horizontal = flip_horizontal
        self.cap = None

        self._initialize_camera()

    def _initialize_camera(self) -> None:
        """Attempts to open the webcam device and configure video properties."""
        print(f"[INFO] Opening webcam (Device Index: {self.camera_index})...")
        
        # Try DirectShow on Windows for faster initialization, fallback to default backend
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"ERROR: Webcam at index {self.camera_index} could not be opened.\n"
                "Please check if your camera is plugged in, permitted, or used by another app."
            )

        # Set requested HD resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

        # Verify actual stream resolution
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[SUCCESS] Camera initialized successfully ({actual_w}x{actual_h}).")

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Reads a frame from the webcam.
        Applies horizontal mirroring if configured.
        Returns (success_flag, bgr_image_array).
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # Fix Left-Right Inverted Camera Feed via Horizontal Mirroring
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        return True, frame

    def is_opened(self) -> bool:
        """Returns True if the camera stream is actively open."""
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        """Releases the camera device hardware resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[INFO] Camera stream released.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
