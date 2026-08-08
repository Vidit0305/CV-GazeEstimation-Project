"""
AI Eye Gaze Tracker - Main Application Entry Point.
Fully local real-time Eye Gaze Estimation system in Python.
"""

import sys
import time
import os
import cv2
import numpy as np

import config
from src.utils import FPSCounter, get_screen_resolution
from src.camera import WebcamStream
from src.face_tracker import FaceTracker
from src.eye_tracker import EyeTracker
from src.iris_tracker import IrisTracker
from src.head_pose import HeadPoseEstimator
from src.calibration import CalibrationManager
from src.gaze_model import GazeRegressor
from src.smoothing import GazeSmoother
from src.visualization import (
    GazeVisualizer,
    DebugHUD,
    compute_gaze_direction,
    compute_confidence
)


def ask_user_calibration_prompt() -> bool:
    """Prompts user whether to load existing calibration or start fresh."""
    if not os.path.exists(config.CALIBRATION_FILE):
        return False

    # Check if stdin is interactive terminal
    if not sys.stdin.isatty():
        print("[INFO] Non-interactive environment detected. Starting new 9-point calibration...")
        return False

    print("\n" + "=" * 55)
    print(" [AI EYE GAZE TRACKER] Calibration Check")
    print(" Existing calibration file found: data/calibration.json")
    print("=" * 55)
    print(" [Y] Load existing calibration")
    print(" [N] Run new 9-point calibration")
    print("-" * 55)

    try:
        choice = input(" Load existing calibration? [Y/N] (default Y): ").strip().lower()
        if choice == 'n':
            return False
        return True
    except (EOFError, KeyboardInterrupt, Exception):
        return False


def main():
    print("=" * 60)
    print("      AI EYE GAZE TRACKER - Real-Time Eye Tracking Model")
    print("=" * 60)

    # 1. Detect Screen Resolution
    screen_w, screen_h = get_screen_resolution()
    print(f"[INFO] Primary Screen Resolution: {screen_w} x {screen_h}")

    # 2. Initialize Hardware & Computer Vision Core Trackers
    try:
        camera = WebcamStream(
            camera_index=config.CAMERA_INDEX,
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
            flip_horizontal=config.FLIP_HORIZONTAL
        )
    except RuntimeError as err:
        print(f"\n[FATAL ERROR] {err}")
        sys.exit(1)

    face_tracker = FaceTracker()
    eye_tracker = EyeTracker()
    iris_tracker = IrisTracker()
    head_pose_estimator = HeadPoseEstimator()

    # 3. Calibration & Model Setup
    calib_manager = CalibrationManager(screen_w=screen_w, screen_h=screen_h)
    gaze_model = GazeRegressor()

    use_existing = ask_user_calibration_prompt()
    dataset = None

    if use_existing:
        dataset = calib_manager.load_calibration_dataset()
        if dataset is None:
            print("[WARNING] Existing calibration data invalid or corrupted. Starting fresh calibration.")
            use_existing = False

    if not use_existing or dataset is None:
        print("\n[INFO] Launching 9-Point Calibration Window...")
        print("[INSTRUCTION] Look directly at each dot as it appears on your screen.\n")
        dataset = calib_manager.run_calibration(
            camera,
            face_tracker,
            eye_tracker,
            iris_tracker,
            head_pose_estimator
        )

    if dataset is None:
        print("[ERROR] Calibration incomplete. Exiting program.")
        camera.release()
        face_tracker.close()
        sys.exit(0)

    features_X, targets_Y = dataset
    gaze_model.train(features_X, targets_Y)
    gaze_model.save()

    # 4. Initialize Smoothing & Visualization Displays
    smoother = GazeSmoother()
    visualizer = GazeVisualizer(screen_w=screen_w, screen_h=screen_h)
    visualizer.init_window()

    debug_hud = DebugHUD()
    fps_counter = FPSCounter()

    window_name = "AI Eye Gaze Tracker - Camera Debug Window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    print("\n" + "=" * 60)
    print(" [STATUS] Gaze Tracking ACTIVE")
    print(" Controls in Camera Debug Window:")
    print("   Q - Quit Application")
    print("   C - Recalibrate 9-Point Grid")
    print("   D - Toggle Diagnostic HUD")
    print("=" * 60 + "\n")

    running = True

    try:
        while running:
            # A. Read webcam frame (mirrored HD feed)
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[WARNING] Empty frame received from webcam. Retrying...")
                time.sleep(0.01)
                continue

            fps = fps_counter.update()

            # B. Process Facial Landmarks
            face_data = face_tracker.process_frame(frame)
            tracking_active = False
            eye_data = None
            iris_data = None
            pose_data = None
            confidence = 0.0
            direction_text = "LOST"

            if face_data is not None:
                lm2d = face_data["landmarks_2d"]
                eye_data = eye_tracker.process_eyes(lm2d)
                iris_data = iris_tracker.process_irises(lm2d)
                pose_data = head_pose_estimator.estimate_pose(lm2d, face_data["frame_size"])

                tracking_active = eye_data["eyes_open"]
                confidence = compute_confidence(tracking_active, eye_data, pose_data)

            # C. Estimate Screen Gaze Position & Filter
            gaze_px = (screen_w // 2, screen_h // 2)

            if tracking_active and iris_data is not None and pose_data is not None:
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

                # Model Prediction
                raw_gaze_x, raw_gaze_y = gaze_model.predict(feature_vec)

                # Signal Noise Filtering (One Euro Filter)
                smooth_gaze_x, smooth_gaze_y = smoother.update(raw_gaze_x, raw_gaze_y)

                gaze_px = (
                    int(smooth_gaze_x * screen_w),
                    int(smooth_gaze_y * screen_h)
                )

                direction_text = compute_gaze_direction(smooth_gaze_x, smooth_gaze_y)

            # D. Render Pygame Screen Visualization Window (Gaze Circle)
            vis_ok = visualizer.render(gaze_px, direction_text, confidence, tracking_active)
            if not vis_ok:
                print("[INFO] User closed gaze screen window.")
                running = False
                break

            # E. Render OpenCV Camera Debug Window
            debug_frame = debug_hud.render_debug_frame(
                frame,
                fps,
                tracking_active,
                eye_data,
                iris_data,
                pose_data,
                direction_text,
                confidence,
                eye_tracker,
                iris_tracker,
                head_pose_estimator
            )

            cv2.imshow(window_name, debug_frame)

            # F. Handle Hotkey Controls
            key = cv2.waitKey(1) & 0xFF
            if key in (config.HOTKEY_QUIT, 27):  # 'q' or ESC
                print("[INFO] Quit key pressed.")
                running = False
                break
            elif key == config.HOTKEY_CALIBRATE:  # 'c'
                print("\n[INFO] Recalibration requested via hotkey...")
                visualizer.close()
                cv2.destroyWindow(window_name)

                smoother.reset()
                dataset = calib_manager.run_calibration(
                    camera,
                    face_tracker,
                    eye_tracker,
                    iris_tracker,
                    head_pose_estimator
                )

                if dataset is not None:
                    features_X, targets_Y = dataset
                    gaze_model.train(features_X, targets_Y)
                    gaze_model.save()

                visualizer.init_window()
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)

            elif key == config.HOTKEY_DEBUG_TOGGLE:  # 'd'
                debug_hud.show_debug_info = not debug_hud.show_debug_info
                print(f"[INFO] Diagnostic HUD toggled: {'ON' if debug_hud.show_debug_info else 'OFF'}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        print("[INFO] Shutting down AI Eye Gaze Tracker...")
        visualizer.close()
        cv2.destroyAllWindows()
        face_tracker.close()
        camera.release()
        print("[SUCCESS] Application closed cleanly.")


if __name__ == "__main__":
    main()
