"""
9-Point Interactive Calibration System.
Displays calibration targets across screen, renders live mini camera preview in the corner,
collects sample 10D feature vectors, performs outlier filtering, and saves calibration dataset.
"""

import time
import math
import cv2
import numpy as np
import pygame
import config
from src.utils import save_json, load_json


class CalibrationManager:
    """Manages the interactive 9-point calibration workflow using Pygame."""

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        samples_per_point: int = config.SAMPLES_PER_POINT,
        calibration_file: str = config.CALIBRATION_FILE
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.samples_per_point = samples_per_point
        self.calibration_file = calibration_file

    def run_calibration(
        self,
        camera,
        face_tracker,
        eye_tracker,
        iris_tracker,
        head_pose_estimator
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Executes interactive 9-point calibration routine with live mini camera feed preview.
        Returns (features_X, targets_Y) arrays, or None if cancelled.
        """
        pygame.init()
        # Set borderless/fullscreen pygame surface
        surface = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.NOFRAME)
        pygame.display.set_caption("AI Eye Gaze Tracker - 9-Point Calibration")
        clock = pygame.time.Clock()

        font_title = pygame.font.SysFont("arial", 32, bold=True)
        font_sub = pygame.font.SysFont("arial", 22)
        font_status = pygame.font.SysFont("consolas", 18)

        collected_features = []
        collected_targets = []

        total_points = len(config.CALIBRATION_GRID)

        print("[CALIBRATION] Starting interactive 9-point calibration with live camera preview...")

        # Mini Camera Preview dimensions (Bottom-Right Corner)
        cam_preview_w = 260
        cam_preview_h = 146
        cam_preview_x = self.screen_w - cam_preview_w - 20
        cam_preview_y = self.screen_h - cam_preview_h - 20

        for idx, (rel_x, rel_y) in enumerate(config.CALIBRATION_GRID, start=1):
            target_x = int(rel_x * self.screen_w)
            target_y = int(rel_y * self.screen_h)

            point_samples = []
            warmup_count = 0
            start_time = time.time()

            while len(point_samples) < self.samples_per_point:
                # Handle Pygame quit events
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                        print("[CALIBRATION] Calibration cancelled by user.")
                        pygame.quit()
                        return None

                # Capture camera frame
                ret, frame = camera.read()
                if not ret or frame is None:
                    continue

                # Extract features & draw debug landmarks on frame for preview
                preview_frame = frame.copy()
                face_data = face_tracker.process_frame(frame)
                feature_vec = None
                tracking_valid = False

                if face_data is not None:
                    lm2d = face_data["landmarks_2d"]
                    eye_data = eye_tracker.process_eyes(lm2d)
                    iris_data = iris_tracker.process_irises(lm2d)
                    pose_data = head_pose_estimator.estimate_pose(lm2d, face_data["frame_size"])

                    # Draw landmarks on mini camera preview
                    eye_tracker.draw_eye_contours(preview_frame, eye_data, color=(0, 255, 200), thickness=1)
                    iris_tracker.draw_irises(preview_frame, iris_data, color=(0, 255, 255), center_color=(0, 0, 255))
                    head_pose_estimator.draw_pose_axes(preview_frame, pose_data)

                    if eye_data["eyes_open"]:
                        l_x = iris_data["left_iris_x"]
                        l_y = iris_data["left_iris_y"]
                        r_x = iris_data["right_iris_x"]
                        r_y = iris_data["right_iris_y"]
                        avg_x = (l_x + r_x) / 2.0
                        avg_y = (l_y + r_y) / 2.0

                        feature_vec = [
                            l_x, l_y,
                            r_x, r_y,
                            avg_x, avg_y,
                            pose_data["yaw"],
                            pose_data["pitch"],
                            pose_data["roll"],
                            eye_data["avg_ear"]
                        ]
                        tracking_valid = True

                # Warmup frames delay per target point
                if tracking_valid:
                    if warmup_count < config.CALIBRATION_WARMUP_FRAMES:
                        warmup_count += 1
                    else:
                        point_samples.append(feature_vec)

                # Render Pygame Calibration Screen
                surface.fill(config.BG_COLOR)

                # Draw Target Point with Pulse Animation
                pulse = math_pulse(time.time() - start_time)
                radius = int(14 + pulse * 6)
                pygame.draw.circle(surface, (0, 200, 255), (target_x, target_y), radius + 8, width=3)
                pygame.draw.circle(surface, (255, 255, 255), (target_x, target_y), radius)
                pygame.draw.circle(surface, (0, 120, 255), (target_x, target_y), 4)

                # Header UI Text
                txt_title = font_title.render("AI Eye Gaze Calibration", True, (240, 240, 250))
                txt_sub = font_sub.render(f"Point {idx} / {total_points} — Look directly at the glowing dot", True, (180, 190, 210))
                
                surface.blit(txt_title, (self.screen_w // 2 - txt_title.get_width() // 2, 40))
                surface.blit(txt_sub, (self.screen_w // 2 - txt_sub.get_width() // 2, 85))

                # Progress Bar
                progress_pct = len(point_samples) / float(self.samples_per_point)
                bar_w, bar_h = 300, 16
                bar_x = self.screen_w // 2 - bar_w // 2
                bar_y = self.screen_h - 80
                
                pygame.draw.rect(surface, (40, 45, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=8)
                if progress_pct > 0:
                    pygame.draw.rect(surface, (0, 220, 180), (bar_x, bar_y, int(bar_w * progress_pct), bar_h), border_radius=8)
                pygame.draw.rect(surface, (100, 110, 130), (bar_x, bar_y, bar_w, bar_h), width=2, border_radius=8)

                # Status info
                status_str = f"Collecting: {len(point_samples)}/{self.samples_per_point}" if tracking_valid else "Face/Eyes Lost - Move into camera view"
                status_color = (0, 230, 180) if tracking_valid else (255, 90, 90)
                txt_status = font_status.render(status_str, True, status_color)
                surface.blit(txt_status, (self.screen_w // 2 - txt_status.get_width() // 2, bar_y - 30))

                # RENDER MINI LIVE CAMERA PREVIEW IN BOTTOM-RIGHT CORNER
                resized_cam = cv2.resize(preview_frame, (cam_preview_w, cam_preview_h))
                cam_rgb = cv2.cvtColor(resized_cam, cv2.COLOR_BGR2RGB)
                # Convert numpy array to Pygame Surface
                cam_surface = pygame.image.frombuffer(cam_rgb.tobytes(), (cam_preview_w, cam_preview_h), "RGB")
                
                # Blit camera preview & draw sleek border
                surface.blit(cam_surface, (cam_preview_x, cam_preview_y))
                pygame.draw.rect(surface, (0, 200, 255), (cam_preview_x, cam_preview_y, cam_preview_w, cam_preview_h), width=2, border_radius=4)
                
                # Camera preview title
                txt_cam_label = font_status.render("LIVE CAMERA", True, (0, 220, 255))
                surface.blit(txt_cam_label, (cam_preview_x + 8, cam_preview_y + 6))

                pygame.display.flip()
                clock.tick(60)

            # Filter Outliers for this point
            filtered_samples = self._filter_outliers(point_samples)
            for feat in filtered_samples:
                collected_features.append(feat)
                collected_targets.append([rel_x, rel_y])

            # Brief pause before next point
            time.sleep(0.15)

        pygame.quit()

        X = np.array(collected_features, dtype=np.float32)
        Y = np.array(collected_targets, dtype=np.float32)

        # Save calibration dataset to JSON
        save_data = {
            "screen_width": self.screen_w,
            "screen_height": self.screen_h,
            "samples_count": len(X),
            "features": X.tolist(),
            "targets": Y.tolist(),
            "timestamp": time.time()
        }
        save_json(save_data, self.calibration_file)
        print(f"[CALIBRATION] Calibration completed with {len(X)} valid samples saved to {self.calibration_file}.")

        return X, Y

    def _filter_outliers(self, samples: list[list[float]]) -> list[list[float]]:
        """Filters feature vectors exceeding standard deviation threshold."""
        if len(samples) < 5:
            return samples

        arr = np.array(samples)
        median = np.median(arr, axis=0)
        std = np.std(arr, axis=0) + 1e-6

        # Standard deviation distance z-score
        z_scores = np.abs((arr - median) / std)
        max_z = np.max(z_scores, axis=1)

        valid_mask = max_z < config.OUTLIER_STD_DEV_THRESHOLD
        filtered = arr[valid_mask].tolist()

        if len(filtered) < 5:
            return samples
        return filtered

    def load_calibration_dataset(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Loads saved calibration feature vectors and targets from JSON."""
        data = load_json(self.calibration_file)
        if data is None or "features" not in data or "targets" not in data:
            return None

        X = np.array(data["features"], dtype=np.float32)
        Y = np.array(data["targets"], dtype=np.float32)

        if len(X) == 0 or len(Y) == 0:
            return None

        # Validate feature vector dimension
        expected_dim = len(config.FEATURE_NAMES)
        if X.ndim < 2 or X.shape[1] != expected_dim:
            print(f"[WARNING] Saved calibration features dimension ({X.shape[1] if X.ndim > 1 else 'invalid'}) does not match current model expected features ({expected_dim}).")
            print("[INFO] Outdated calibration file will be ignored and replaced with fresh calibration.")
            return None

        return X, Y


def math_pulse(t: float) -> float:
    """Returns a smooth sine pulse factor between 0.0 and 1.0."""
    return 0.5 + 0.5 * np.sin(t * 6.0)
