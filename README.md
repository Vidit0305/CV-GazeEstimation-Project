# AI Eye Gaze Tracker

A real-time, fully local **Computer Vision and Machine Learning Eye Gaze Estimation System** built in Python. 

The application utilizes your laptop's built-in webcam to track facial landmarks, iris center offsets, and 3D head pose in real time. Following an interactive **9-point calibration routine**, a non-linear regression model maps your feature vectors to your computer screen coordinates, animating a glowing target dot that smoothly follows where you look.

---

## 🌟 Key Features

* **Real-Time Webcam Processing**: Optimized for local laptop CPU execution (targeting 20–30+ FPS at 640x480 resolution).
* **Refined Iris Tracking**: Calculates scale-invariant relative iris position `(left_iris_x, left_iris_y, right_iris_x, right_iris_y)` relative to eye corners and eyelids using vector projections.
* **3D Head Pose Estimation**: Estimates Head Yaw, Pitch, and Roll angles via OpenCV `solvePnP` to compensate for head movement during gaze estimation.
* **Interactive 9-Point Calibration Grid**: Displays full-screen target points, collects sample feature vectors, performs statistical outlier filtering, and saves calibration to `data/calibration.json`.
* **Polynomial Ridge Regression Model**: Lightweight non-linear model mapping 7D eye + head features to normalized screen coordinates `(gaze_x, gaze_y)`.
* **One Euro Signal Filtering**: Advanced low-pass filter (1€ Filter) that eliminates micro-saccadic eye jitter when stationary while maintaining zero lag during fast gaze movements.
* **Dual Visualization UI**:
  * **Screen Gaze Window** (Pygame): High-performance screen canvas with a glowing target dot following estimated gaze.
  * **Camera Diagnostic HUD** (OpenCV): Live camera feed with facial mesh overlay, iris rings, 3D head pose tripod, FPS, confidence %, and gaze direction indicators.
* **Local Storage & Persistence**: Automatically saves and reloads calibration data without requiring cloud services or databases.

---

## 🛠️ Technology Stack

* **Python 3.10+**
* **OpenCV (`opencv-python`)**: Video capture, frame preprocessing, head pose `solvePnP`, HUD overlay rendering.
* **MediaPipe (`mediapipe.solutions.face_mesh`)**: 478 3D facial landmarks with refined iris landmark tracking.
* **Scikit-Learn (`scikit-learn`)**: Polynomial feature transformation and Ridge regression mapping.
* **Pygame (`pygame`)**: Interactive calibration target display and screen gaze circle visualization.
* **NumPy (`numpy`)**: Fast array operations, matrix projections, and statistical outlier filtering.

---

## 📁 Project Structure

```text
CV-GazeEstimation-Project/
│
├── main.py                # Main application orchestrator & real-time loop
├── config.py              # System parameters, camera settings, calibration grid
├── requirements.txt       # Python library dependencies
├── README.md              # Project documentation
│
├── src/
│   ├── __init__.py
│   ├── camera.py          # OpenCV WebcamStream wrapper & hardware error handling
│   ├── face_tracker.py    # MediaPipe FaceMesh landmark processor
│   ├── eye_tracker.py     # Eye contour extraction & Eye Aspect Ratio (EAR)
│   ├── iris_tracker.py    # Relative iris coordinate vector projection
│   ├── head_pose.py       # 3D head pose estimation (solvePnP Yaw, Pitch, Roll)
│   ├── calibration.py     # Interactive 9-point calibration routine & storage
│   ├── gaze_model.py      # Polynomial Ridge Regression model (train/predict)
│   ├── smoothing.py       # One Euro Filter (1€ Filter) & GazeSmoother
│   ├── visualization.py   # Pygame GazeVisualizer & OpenCV DebugHUD
│   └── utils.py           # FPSCounter, screen resolution detector, JSON I/O
│
├── data/
│   └── calibration.json   # Saved calibration feature vectors & targets
│
└── models/
    └── gaze_model.pkl     # Serialized trained gaze regression pipeline
```

---

## 🚀 Installation & Quickstart

### 1. Clone or Open Project
Navigate to the project directory:
```bash
cd CV-GazeEstimation-Project
```

### 2. Install Dependencies
Install required packages via `pip`:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Launch the gaze tracker:
```bash
python main.py
```

---

## 🎯 How Calibration Works

1. On launch, if `data/calibration.json` exists, you will be asked if you want to load existing calibration or run a new calibration.
2. When starting a **new calibration**, a full-screen dark window appears displaying a glowing target dot.
3. The target dot moves through **9 key screen positions**:
   * Top-Left, Top-Center, Top-Right
   * Middle-Left, Center, Middle-Right
   * Bottom-Left, Bottom-Center, Bottom-Right
4. **Look directly at the center of the target dot** while the progress bar fills (`Collecting samples: ████████░░`).
5. The system collects multiple sample frames per point, applies statistical outlier rejection to eliminate blinks or momentary head shifts, and fits the Polynomial Ridge Regression model.

---

## 🎮 Controls & Keyboard Shortcuts

When the OpenCV Camera Debug Window is active:

| Key | Action |
|---|---|
| `Q` / `ESC` | **Quit** the application cleanly |
| `C` | Trigger **Recalibration** routine (launches 9-point grid) |
| `D` | Toggle **Diagnostic HUD** overlay in camera window |

---

## 🔬 Mathematical Architecture

1. **Iris Relative Coordinates**:
   $$\text{iris}_x = \frac{\mathbf{v}_{\text{iris}} \cdot \mathbf{u}_{\text{horizontal}}}{L_{\text{horizontal}}}, \quad \text{iris}_y = \frac{\mathbf{v}_{\text{iris}} \cdot \mathbf{u}_{\text{vertical}}}{L_{\text{vertical}}}$$
2. **Feature Vector (7D)**:
   $$\mathbf{X} = [\text{left\_iris\_x}, \text{left\_iris\_y}, \text{right\_iris\_x}, \text{right\_iris\_y}, \text{head\_yaw}, \text{head\_pitch}, \text{head\_roll}]$$
3. **Regression Pipeline**:
   $$\hat{\mathbf{Y}}_{\text{screen}} = \text{Ridge}(\text{StandardScaler}(\text{PolynomialFeatures}(\mathbf{X}, \text{degree}=2)))$$
4. **Adaptive Noise Filtering (1€ Filter)**:
   $$\alpha = \frac{2\pi f_c \Delta t}{2\pi f_c \Delta t + 1}, \quad f_c = f_{\text{min}} + \beta |\dot{x}|$$

---

## 🔮 Future Roadmap (v2 / v3)

* **Version 2**: Enhanced multi-point dynamic polynomial smoothing.
* **Version 3**: Optional OS Mouse Cursor control mode.
* **Version 4**: Eye Blink & Dwell time gesture detection for mouse clicking (`Blink` = Click, `Dwell` = Hover).

---

## 📄 License

Developed as a local Computer Vision / Machine Learning demonstration project.
