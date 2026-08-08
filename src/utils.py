"""
Utility Functions and Helpers for AI Eye Gaze Tracker.
"""

import time
import json
import os
import cv2
import numpy as np


class FPSCounter:
    """Calculates and tracks frames per second (FPS)."""

    def __init__(self, buffer_size: int = 30):
        self.buffer_size = buffer_size
        self.frame_times = []
        self.last_time = time.time()
        self.fps = 0.0

    def update(self) -> float:
        """Call on each frame to update FPS calculation."""
        current_time = time.time()
        delta = current_time - self.last_time
        self.last_time = current_time

        if delta > 0:
            self.frame_times.append(delta)
            if len(self.frame_times) > self.buffer_size:
                self.frame_times.pop(0)

            avg_delta = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_delta if avg_delta > 0 else 0.0

        return self.fps

    def get_fps(self) -> float:
        """Returns the current FPS."""
        return self.fps


def get_screen_resolution() -> tuple[int, int]:
    """
    Detects the primary display monitor resolution.
    Uses tkinter or fallback defaults.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass

    # Pygame fallback
    try:
        import pygame
        pygame.init()
        info = pygame.display.Info()
        width, height = info.current_w, info.current_h
        pygame.quit()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass

    # Standard desktop default fallback
    return 1920, 1080


def save_json(data: dict | list, file_path: str) -> bool:
    """Saves data dictionary to a JSON file safely."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save JSON to {file_path}: {e}")
        return False


def load_json(file_path: str) -> dict | list | None:
    """Loads JSON data from file."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load JSON from {file_path}: {e}")
        return None


def draw_text_with_bg(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (20, 20, 25),
    thickness: int = 1,
    padding: int = 4
) -> None:
    """Renders text with a semi-opaque rounded rectangular background box."""
    x, y = pos
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    bg_rect_min = (x - padding, y - text_h - padding)
    bg_rect_max = (x + text_w + padding, y + baseline + padding)
    
    cv2.rectangle(img, bg_rect_min, bg_rect_max, bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
