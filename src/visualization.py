"""
Visualization and Debug HUD Module.
Provides dual visual displays:
1. GazeVisualizer: Transparent desktop overlay Pygame window showing gaze circle movement,
   gaze trails, optional live camera Picture-in-Picture (PiP), and recording status.
2. DebugHUD: OpenCV webcam feed overlay showing facial landmarks, head pose axes, FPS,
   recording status, and tracking metrics.
"""

import os
import sys
import time
import ctypes
import cv2
import numpy as np
import pygame
import config
from src.utils import draw_text_with_bg

# Transparent ColorKey RGB (mapped to 100% invisible on Windows)
TRANSPARENT_COLORKEY = (1, 1, 1)

# Win32 Constants for Always-On-Top Layered Window
HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
SWP_FRAMECHANGED = 0x0020
WDA_EXCLUDEFROMCAPTURE = 0x00000011


def set_window_transparent_and_topmost(screen_w: int, screen_h: int, click_through: bool = True) -> int | None:
    """
    Configures Pygame window on Windows to be 100% transparent background,
    pinned to HWND_TOPMOST above all opened apps/browsers, tool-window (no taskbar clutter),
    fully click-through, and excluded from raw capture so screen recordings record real desktop content.
    """
    if sys.platform != "win32":
        return None
    try:
        hwnd = pygame.display.get_wm_info().get("window")
        if not hwnd:
            return None

        GWL_STYLE = -16
        GWL_EXSTYLE = -20
        WS_POPUP = 0x80000000
        WS_VISIBLE = 0x10000000

        WS_EX_LAYERED = 0x00080000
        WS_EX_TOPMOST = 0x00000008
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_NOACTIVATE = 0x08000000
        LWA_COLORKEY = 0x00000001

        # Standard window style: popup + visible
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style | WS_POPUP | WS_VISIBLE)

        # Extended window style: layered + topmost + toolwindow + noactivate + transparent
        ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        new_ex_style = ex_style | WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        if click_through:
            new_ex_style |= WS_EX_TRANSPARENT

        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_ex_style)
        # COLORREF 0x00010101 maps to RGB (1, 1, 1) for transparency colorkey
        ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0x00010101, 0, LWA_COLORKEY)

        # Exclude transparent colorkey window from screen capture so recording captures the real desktop apps
        try:
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        except Exception:
            pass

        # Force window to top of all desktop windows and position at (0,0) with exact full screen size
        ctypes.windll.user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            0, 0, screen_w, screen_h,
            SWP_SHOWWINDOW | SWP_NOACTIVATE | SWP_FRAMECHANGED
        )
        return hwnd
    except Exception as e:
        print(f"[WARNING] Could not configure transparent desktop overlay: {e}")
        return None


class GazeVisualizer:
    """Renders transparent screen gaze target overlay, gaze trails, and Camera PiP directly on desktop."""

    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.surface = None
        self.hwnd = None
        self.is_active = False
        self.render_frame_count = 0

        # Gaze trail history
        self.gaze_trail = []
        self.max_trail = config.GAZE_TRAIL_LENGTH

        # Grace period memory for momentary eye blinks
        self.last_gaze_px = (screen_w // 2, screen_h // 2)
        self.lost_frames_count = 0
        self.max_grace_frames = 15

        # Picture-in-Picture (PiP) settings
        self.show_camera_pip = config.ENABLE_CAMERA_PIP
        self.pip_w = config.PIP_WIDTH
        self.pip_h = config.PIP_HEIGHT
        self.pip_pad = config.PIP_PADDING
        self.pip_pos = config.PIP_POSITION

        self.font_rec = None
        self.font_pip = None

    def init_window(self) -> None:
        """Initializes Pygame transparent desktop overlay surface."""
        if not self.is_active:
            os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
            os.environ["SDL_HINT_ALLOW_TOPMOST"] = "1"
            os.environ["SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"
            os.environ["SDL_HINT_MOUSE_FOCUS_CLICKTHROUGH"] = "1"

            pygame.init()
            self.surface = pygame.display.set_mode(
                (self.screen_w, self.screen_h),
                pygame.NOFRAME | pygame.DOUBLEBUF
            )
            pygame.display.set_caption("AI Eye Gaze Tracker - Transparent Desktop Overlay")

            # Enable Windows desktop window transparency & lock topmost z-order
            self.hwnd = set_window_transparent_and_topmost(self.screen_w, self.screen_h, click_through=True)

            self.font_rec = pygame.font.SysFont("arial", 15, bold=True)
            self.font_pip = pygame.font.SysFont("consolas", 12, bold=True)
            self.is_active = True

    def toggle_pip(self) -> bool:
        """Toggles camera Picture-in-Picture overlay on desktop."""
        self.show_camera_pip = not self.show_camera_pip
        print(f"[INFO] Camera Picture-in-Picture overlay: {'ENABLED' if self.show_camera_pip else 'DISABLED'}")
        return self.show_camera_pip

    def render(
        self,
        gaze_px: tuple[int, int],
        direction_text: str,
        confidence: float,
        tracking_active: bool,
        camera_frame: np.ndarray | None = None,
        is_recording: bool = False,
        rec_duration_str: str = "00:00"
    ) -> bool:
        """
        Renders gaze circle target, smooth trails, camera PiP, and recording badge
        directly over your laptop desktop screen.
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
                elif event.key == pygame.K_p:
                    self.toggle_pip()

        # Periodically enforce HWND_TOPMOST so overlay never gets obscured behind active apps/browsers
        self.render_frame_count += 1
        if sys.platform == "win32" and self.hwnd and (self.render_frame_count % 15 == 0):
            try:
                ctypes.windll.user32.SetWindowPos(
                    self.hwnd,
                    HWND_TOPMOST,
                    0, 0, 0, 0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE
                )
            except Exception:
                pass

        # Fill with Transparent ColorKey (Makes background 100% invisible on desktop)
        self.surface.fill(TRANSPARENT_COLORKEY)

        x, y = gaze_px
        x = int(np.clip(x, 15, self.screen_w - 15))
        y = int(np.clip(y, 15, self.screen_h - 15))

        # Update Gaze Position with Blink Grace Period
        show_indicator = False
        if tracking_active:
            self.last_gaze_px = (x, y)
            self.lost_frames_count = 0
            show_indicator = True
        else:
            self.lost_frames_count += 1
            if self.lost_frames_count <= self.max_grace_frames:
                # Hold last known position smoothly during momentary blinks/head shifts
                x, y = self.last_gaze_px
                show_indicator = True

        # Update Trail Buffer
        if show_indicator:
            self.gaze_trail.append((x, y))
            if len(self.gaze_trail) > self.max_trail:
                self.gaze_trail.pop(0)
        else:
            if self.gaze_trail:
                self.gaze_trail.pop(0)

        # 1. Draw Smooth Gaze Trails
        if len(self.gaze_trail) > 1:
            for i in range(len(self.gaze_trail) - 1):
                pt1 = self.gaze_trail[i]
                pt2 = self.gaze_trail[i + 1]
                factor = (i + 1) / float(len(self.gaze_trail))
                trail_c = (
                    int(0 * factor + 100 * (1 - factor)),
                    int(220 * factor),
                    int(255 * factor)
                )
                width = int(1 + 3 * factor)
                pygame.draw.line(self.surface, trail_c, pt1, pt2, width)

        # 2. Draw Gaze Target Indicator
        if show_indicator:
            # Outer cyan glow ring
            pygame.draw.circle(self.surface, (0, 180, 255), (x, y), config.DOT_RADIUS + 6, width=2)
            # Main cyan dot
            pygame.draw.circle(self.surface, config.DOT_COLOR, (x, y), config.DOT_RADIUS)
            # Center bright dot
            pygame.draw.circle(self.surface, config.DOT_BORDER_COLOR, (x, y), 4)

            # Crosshairs
            line_color = (255, 255, 255)
            pygame.draw.line(self.surface, line_color, (x - 18, y), (x - 6, y), 2)
            pygame.draw.line(self.surface, line_color, (x + 6, y), (x + 18, y), 2)
            pygame.draw.line(self.surface, line_color, (x, y - 18), (x, y - 6), 2)
            pygame.draw.line(self.surface, line_color, (x, y + 6), (x, y + 18), 2)

        # 3. Draw Live Camera Picture-in-Picture (PiP) on Desktop
        if self.show_camera_pip and camera_frame is not None:
            self._render_camera_pip_overlay(camera_frame, tracking_active, confidence)

        # 4. Draw Recording Status Badge on Desktop
        if is_recording:
            self._render_rec_badge(rec_duration_str)

        pygame.display.flip()
        return True

    def _render_camera_pip_overlay(
        self,
        camera_frame: np.ndarray,
        tracking_active: bool,
        confidence: float
    ) -> None:
        """Renders live webcam feed inside desktop overlay surface."""
        pw, ph = self.pip_w, self.pip_h
        pad_x = self.pip_pad
        pad_y = self.pip_pad + 35  # Account for taskbar at bottom

        if self.pip_pos == "bottom_right":
            px, py = self.screen_w - pw - pad_x, self.screen_h - ph - pad_y
        elif self.pip_pos == "top_right":
            px, py = self.screen_w - pw - pad_x, pad_x + 40
        elif self.pip_pos == "bottom_left":
            px, py = pad_x, self.screen_h - ph - pad_y
        else:
            px, py = pad_x, pad_x + 40

        try:
            # Resize and convert OpenCV BGR frame to Pygame RGB
            resized_cam = cv2.resize(camera_frame, (pw, ph), interpolation=cv2.INTER_AREA)
            cam_rgb = cv2.cvtColor(resized_cam, cv2.COLOR_BGR2RGB)
            cam_surface = pygame.image.frombuffer(cam_rgb.tobytes(), (pw, ph), "RGB")

            # Draw card header backdrop
            header_rect = pygame.Rect(px - 2, py - 20, pw + 4, 20)
            pygame.draw.rect(self.surface, (20, 26, 36), header_rect)

            # Blit camera image
            self.surface.blit(cam_surface, (px, py))

            # Border
            border_color = (0, 230, 140) if tracking_active else (220, 70, 70)
            pygame.draw.rect(self.surface, border_color, pygame.Rect(px, py, pw, ph), 2)
            pygame.draw.rect(self.surface, (70, 90, 120), header_rect, 1)

            # Header text
            if self.font_pip:
                title_surf = self.font_pip.render("WEBCAM [P]", True, (0, 220, 255))
                self.surface.blit(title_surf, (px + 4, py - 17))

                stat_text = f"{confidence:.0f}%" if tracking_active else "LOST"
                stat_surf = self.font_pip.render(stat_text, True, border_color)
                self.surface.blit(stat_surf, (px + pw - stat_surf.get_width() - 4, py - 17))
        except Exception:
            pass

    def _render_rec_badge(self, rec_duration_str: str) -> None:
        """Renders live recording indicator at top-left of desktop."""
        bx, by, bw, bh = 20, 20, 160, 30

        # Background badge
        badge_rect = pygame.Rect(bx, by, bw, bh)
        pygame.draw.rect(self.surface, (18, 22, 30), badge_rect)
        pygame.draw.rect(self.surface, (70, 85, 110), badge_rect, 1)

        # Pulsing red circle
        pulse = (int(time.time() * 2) % 2) == 0
        dot_c = (255, 40, 40) if pulse else (180, 20, 20)
        pygame.draw.circle(self.surface, dot_c, (bx + 16, by + bh // 2), 6)
        pygame.draw.circle(self.surface, (255, 255, 255), (bx + 16, by + bh // 2), 7, 1)

        # Text
        if self.font_rec:
            rec_surf = self.font_rec.render(f"REC  {rec_duration_str}", True, (240, 245, 255))
            self.surface.blit(rec_surf, (bx + 30, by + 6))

    def close(self) -> None:
        """Closes Pygame visualization window."""
        if self.is_active:
            pygame.quit()
            self.is_active = False


class DebugHUD:
    """Manages OpenCV webcam overlay visualization and diagnostic HUD panel."""

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
        head_pose_estimator,
        is_recording: bool = False,
        rec_duration_str: str = "00:00",
        pip_active: bool = True
    ) -> np.ndarray:
        """Draws landmarks, head pose vectors, recording badge, and stats HUD overlay on camera frame."""
        output_frame = frame.copy()

        # Draw Landmarks if tracking active
        if tracking_active and eye_data is not None and iris_data is not None:
            eye_tracker.draw_eye_contours(output_frame, eye_data, color=(0, 255, 200), thickness=1)
            iris_tracker.draw_irises(output_frame, iris_data, color=(0, 255, 255), center_color=(0, 0, 255))
            if pose_data is not None:
                head_pose_estimator.draw_pose_axes(output_frame, pose_data)

        # Draw Compact HUD Panel if enabled
        if self.show_debug_info:
            self._draw_status_panel(
                output_frame,
                fps,
                tracking_active,
                eye_data,
                iris_data,
                direction_text,
                confidence,
                is_recording,
                rec_duration_str,
                pip_active
            )

        return output_frame

    def _draw_status_panel(
        self,
        frame: np.ndarray,
        fps: float,
        tracking_active: bool,
        eye_data: dict | None,
        iris_data: dict | None,
        direction_text: str,
        confidence: float,
        is_recording: bool,
        rec_duration_str: str,
        pip_active: bool
    ) -> None:
        """Renders small, sleek diagnostic HUD badge on top-left of webcam view."""
        h, w, _ = frame.shape
        panel_w = 210
        panel_h = 172

        # Semi-transparent background card
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (12, 15, 22), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Border
        cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h), (50, 65, 85), 1)

        # Header Title
        cv2.putText(frame, "AI EYE GAZE TRACKER", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (16, 28), (8 + panel_w - 8, 28), (40, 55, 75), 1)

        # Status indicators
        face_status = "OK" if tracking_active else "MISSING"
        iris_status = "OK" if (tracking_active and iris_data) else "LOST"
        track_status = "ACTIVE" if tracking_active else "LOST"
        track_color = (0, 230, 140) if tracking_active else (50, 50, 255)

        rec_status = f"REC [{rec_duration_str}]" if is_recording else "IDLE"
        rec_color = (40, 40, 255) if is_recording else (140, 150, 165)

        pip_status = "ON (Desktop)" if pip_active else "OFF"
        pip_color = (0, 220, 255) if pip_active else (140, 150, 165)

        lines = [
            (f"FPS: {fps:.1f}", (220, 225, 235)),
            (f"Face/Iris: {face_status}/{iris_status}", (220, 225, 235)),
            (f"Gaze: {direction_text}", (220, 225, 235)),
            (f"Confidence: {confidence:.0f}%", (220, 225, 235)),
            (f"Status: {track_status}", track_color),
            (f"Video REC: {rec_status}", rec_color),
            (f"Camera PiP: {pip_status}", pip_color)
        ]

        y = 44
        for text, color in lines:
            cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
            y += 18

        # Footer guide
        draw_text_with_bg(
            frame,
            "[Q] Quit | [C] Recalib | [D] HUD | [R] Record | [P] PiP",
            (8, h - 12),
            font_scale=0.38,
            text_color=(200, 215, 240),
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

