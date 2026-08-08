"""
AI Eye Gaze Tracker Configuration
Contains system constants, threshold settings, HD resolution, and model configurations.
"""

import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# File Paths
CALIBRATION_FILE = os.path.join(DATA_DIR, "calibration.json")
MODEL_FILE = os.path.join(MODELS_DIR, "gaze_model.pkl")

# Camera Settings (HD Resolution for High Quality)
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30
FLIP_HORIZONTAL = True  # Fix Left-Right Inverted Mirror Camera Feed

# MediaPipe Settings
MAX_NUM_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Calibration Grid Points (Normalized Screen Coordinates [0.0 - 1.0])
CALIBRATION_GRID = [
    (0.1, 0.1),  # Top-Left
    (0.5, 0.1),  # Top-Center
    (0.9, 0.1),  # Top-Right
    (0.1, 0.5),  # Middle-Left
    (0.5, 0.5),  # Center
    (0.9, 0.5),  # Middle-Right
    (0.1, 0.9),  # Bottom-Left
    (0.5, 0.9),  # Bottom-Center
    (0.9, 0.9),  # Bottom-Right
]

SAMPLES_PER_POINT = 40
CALIBRATION_WARMUP_FRAMES = 12
OUTLIER_STD_DEV_THRESHOLD = 1.8

# Feature Vector Dimension & Names (10D Rich Vector)
FEATURE_NAMES = [
    "left_iris_x",
    "left_iris_y",
    "right_iris_x",
    "right_iris_y",
    "avg_iris_x",
    "avg_iris_y",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "avg_ear"
]

# One Euro Filter (Smoothing) Settings - Tuned for Buttery Smooth, Calmed Motion
ONE_EURO_MIN_CUTOFF = 0.05   # Lower cutoff eliminates jitter/shaking when looking at a spot
ONE_EURO_BETA = 0.05         # Speed factor for rapid lag-free movements
ONE_EURO_D_CUTOFF = 1.0

# Visualization Settings
DOT_RADIUS = 14
DOT_COLOR = (0, 230, 255)       # Cyan accent
DOT_BORDER_COLOR = (255, 255, 255)
BG_COLOR = (12, 14, 20)          # Dark modern theme

# Debug Window Controls
HOTKEY_QUIT = ord('q')
HOTKEY_CALIBRATE = ord('c')
HOTKEY_DEBUG_TOGGLE = ord('d')
