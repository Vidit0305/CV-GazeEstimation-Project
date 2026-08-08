"""
Visualization and Debug HUD Module.
Provides dual visual displays:
1. GazeVisualizer: Transparent desktop overlay Pygame window showing gaze circle movement over your real laptop screen.
2. DebugHUD: OpenCV webcam feed overlay showing facial landmarks, head pose axes, FPS, and tracking metrics.
"""

import sys
import ctypes
import cv2
import numpy as np
import pygame
import config
from src.utils import draw_text_with_bg

# Transparent ColorKey RGB (mapped to 100% invisible on Windows)
TRANSPARENT_COLORKEY = (1, 1, 1)


def set_window_transparent_and_topmost(click_through: bool = True) -> bool:
    """
    Configures Pygame window on Windows to be 100% transparent background,
    top-most above all desktop windows, and optionally click-through.
    """
    if sys.platform != "win32":
        return False
    try:
        hwnd = pygame.display.get_wm_info().get("window")
        if not hwnd:
            return False

        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x80000
        WS_EX_TOPMOST = 0x8
        WS_EX_TRANSPARENT = 0x20
        LWA_COLORKEY = 0x1

        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        new_style = style | WS_EX_LAYERED | WS_EX_TOPMOST
        if click_through:
            new_style |= WS_EX_TRANSPARENT

        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
        # COLORREF 0x00010101 maps to RGB (1, 1, 1)
        ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0x00010101, 0, LWA_COLORKEY)
        return True
    except Exception as e:
        print(f"[WARNING] Could not configure transparent desktop overlay: {e}")
        return False


class GazeVisualizer:
    """Renders transparent screen gaze target overlay directly on top of your laptop desktop."""

    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.surface = None
        self.is_active = False

    def init_window(self) -> None:
        """Initializes Pygame transparent desktop overlay surface."""
        if not self.is_active:
            pygame.init()
            self.surface = pygame.display.set_mode(
                (self.screen_w, self.screen_h),
                pygame.NOFRAME | pygame.DOUBLEBUF
            )
            pygame.display.set_caption("AI Eye Gaze Tracker - Transparent Overlay")

            # Enable Windows desktop window transparency & click-through
            set_window_transparent_and_topmost(click_through=True)

            self.font_dir = pygame.font.SysFont("arial", 20, bold=True)
            self.font_info = pygame.font.SysFont("consolas", 14)
            self.is_active = True

    def render(
        self,
        gaze_px: tuple[int, int],
        direction_text: str,
        confidence: float,
        tracking_active: bool
    ) -> bool:
        """
        Renders gaze circle target directly over your laptop desktop screen.
        Returns False if user requested quit.
        """
        if not self.is_active or self.surface is None:
            return True

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False

        # Fill with Transparent ColorKey (Makes background 100% invisible on desktop)
        self.surface.fill(TRANSPARENT_COLORKEY)

        x, y = gaze_px
        x = int(np.clip(x, 15, self.screen_w - 15))
        y = int(np.clip(y, 15, self.screen_h - 15))

        if tracking_active:
            # Draw Gaze Circle Indicator over your desktop
            pygame.draw.circle(self.surface, (0, 180, 255), (x, y), config.DOT_RADIUS + 6, width=2)
            pygame.draw.circle(self.surface, config.DOT_COLOR, (x, y), config.DOT_RADIUS)
            pygame.draw.circle(self.surface, config.DOT_BORDER_COLOR, (x, y), 4)

            # Draw crosshair lines
            line_color = (255, 255, 255)
            pygame.draw.line(self.surface, line_color, (x - 18, y), (x + 18, y), 1)
            pygame.draw.line(self.surface, line_color, (x, y - 18), (x, y + 18), 1)

        pygame.display.flip()
        return True

    def close(self) -> None:
        """Closes Pygame visualization window."""
        if self.is_active:
            pygame.quit()
            self.is_active = False


class DebugHUD:
    """Manages OpenCV webcam overlay visualization and compact diagnostic HUD panel."""

    def __init__(self):
        self.show_debug_info = True

    def render_debug_frame(
        self,
        frame: np.ndarray,
        fps: float,
        tracking_active: bool,
        eye_data: dict | None,
        iris_data: dict | None,
        pose_data: dict | None,
        direction_text: str,
        confidence: float,
        eye_tracker,
        iris_tracker,
        head_pose_estimator
    ) -> np.ndarray:
        """Draws landmarks, head pose vectors, and stats HUD overlay on camera frame."""
        output_frame = frame.copy()

        # Draw Landmarks if tracking active
        if tracking_active and eye_data is not None and iris_data is not None:
            eye_tracker.draw_eye_contours(output_frame, eye_data, color=(0, 255, 200), thickness=1)
            iris_tracker.draw_irises(output_frame, iris_data, color=(0, 255, 255), center_color=(0, 0, 255))
            if pose_data is not None:
                head_pose_estimator.draw_pose_axes(output_frame, pose_data)

        # Draw Compact HUD Panel if enabled
        if self.show_debug_info:
            self._draw_status_panel(output_frame, fps, tracking_active, eye_data, iris_data, direction_text, confidence)

        return output_frame

    def _draw_status_panel(
        self,
        frame: np.ndarray,
        fps: float,
        tracking_active: bool,
        eye_data: dict | None,
        iris_data: dict | None,
        direction_text: str,
        confidence: float
    ) -> None:
        """Renders small, sleek diagnostic HUD badge on top-left of webcam view."""
        h, w, _ = frame.shape
        panel_w = 190
        panel_h = 135

        # Semi-transparent background card
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (12, 15, 22), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Border
        cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h), (50, 65, 85), 1)

        # Header Title
        cv2.putText(frame, "AI EYE GAZE", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (16, 28), (8 + panel_w - 8, 28), (40, 55, 75), 1)

        # Status indicators
        face_status = "OK" if tracking_active else "MISSING"
        iris_status = "OK" if (tracking_active and iris_data) else "LOST"
        track_status = "ACTIVE" if tracking_active else "LOST"
        track_color = (0, 230, 140) if tracking_active else (50, 50, 255)

        lines = [
            f"FPS: {fps:.1f}",
            f"Face/Iris: {face_status}/{iris_status}",
            f"Gaze: {direction_text}",
            f"Confidence: {confidence:.0f}%",
            f"Status: {track_status}"
        ]

        y = 44
        for line in lines:
            color = (220, 225, 235)
            if "Status:" in line:
                color = track_color
            cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            y += 18

        # Footer guide
        draw_text_with_bg(
            frame,
            "[Q] Quit | [C] Recalibrate | [D] HUD",
            (8, h - 12),
            font_scale=0.40,
            text_color=(200, 210, 230),
            bg_color=(15, 20, 30)
        )


def compute_gaze_direction(gaze_x: float, gaze_y: float) -> str:
    """Categorizes normalized gaze position into 9 directional zones."""
    horizontal = "CENTER"
    if gaze_x < 0.38:
        horizontal = "LEFT"
    elif gaze_x > 0.62:
        horizontal = "RIGHT"

    vertical = ""
    if gaze_y < 0.38:
        vertical = "UP"
    elif gaze_y > 0.62:
        vertical = "DOWN"

    if vertical and horizontal != "CENTER":
        return f"{vertical}-{horizontal}"
    elif vertical:
        return vertical
    return horizontal


def compute_confidence(
    tracking_active: bool,
    eye_data: dict | None,
    pose_data: dict | None
) -> float:
    """Calculates approximate tracking quality confidence percentage."""
    if not tracking_active or eye_data is None:
        return 0.0

    conf = 98.0

    # Penalty for closed/blinking eyes
    if eye_data.get("left_closed", False):
        conf -= 35.0
    if eye_data.get("right_closed", False):
        conf -= 35.0

    # Penalty for extreme head pose
    if pose_data is not None:
        yaw = abs(pose_data.get("yaw", 0.0))
        pitch = abs(pose_data.get("pitch", 0.0))

        if yaw > 30.0:
            conf -= (yaw - 30.0) * 1.2
        if pitch > 25.0:
            conf -= (pitch - 25.0) * 1.5

    return float(np.clip(conf, 0.0, 100.0))
