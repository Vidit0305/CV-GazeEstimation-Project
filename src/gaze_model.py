"""
Gaze Regression Model.
Maps 10D Eye + Head feature vectors to 2D normalized screen coordinates (gaze_x, gaze_y).
Uses Polynomial Ridge Regression for smooth, highly accurate, non-linear calibration mapping.
"""

import pickle
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
import config


class GazeRegressor:
    """Regression model mapping facial/eye features to screen coordinates."""

    def __init__(self, degree: int = 2, alpha: float = 0.05):
        self.degree = degree
        self.alpha = alpha
        self.is_trained = False
        self.model = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        """Constructs Polynomial Ridge Regression pipeline."""
        return Pipeline([
            ("poly", PolynomialFeatures(degree=self.degree, include_bias=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha))
        ])

    def train(self, features: list[list[float]] | np.ndarray, targets: list[list[float]] | np.ndarray) -> float:
        """
        Trains the regression model on calibration dataset.
        features shape: (N, 10)
        targets shape: (N, 2) normalized [0.0 - 1.0]
        Returns R^2 training score.
        """
        X = np.array(features, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)

        if len(X) < 10:
            raise ValueError(f"Insufficient calibration samples ({len(X)} samples). Need at least 10.")

        self.model = self._build_pipeline()
        self.model.fit(X, y)
        self.is_trained = True

        r2_score = float(self.model.score(X, y))
        print(f"[MODEL] Gaze Regression model trained successfully. R^2 score: {r2_score:.4f}")
        return r2_score

    def predict(self, feature_vector: list[float] | np.ndarray) -> tuple[float, float]:
        """
        Predicts normalized screen gaze position (gaze_x, gaze_y) [0.0 - 1.0].
        """
        if not self.is_trained:
            return 0.5, 0.5

        X_input = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        pred = self.model.predict(X_input)[0]

        # Clamp prediction within normalized screen boundaries
        gaze_x = float(np.clip(pred[0], 0.0, 1.0))
        gaze_y = float(np.clip(pred[1], 0.0, 1.0))

        return gaze_x, gaze_y

    def save(self, file_path: str = config.MODEL_FILE) -> bool:
        """Saves trained model state to disk."""
        if not self.is_trained:
            print("[WARNING] Cannot save model: model is not trained yet.")
            return False
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump(self.model, f)
            print(f"[INFO] Gaze model saved to {file_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save gaze model: {e}")
            return False

    def load(self, file_path: str = config.MODEL_FILE) -> bool:
        """Loads trained model state from disk."""
        if not os.path.exists(file_path):
            return False
        try:
            with open(file_path, "rb") as f:
                loaded_model = pickle.load(f)

            expected_dim = len(config.FEATURE_NAMES)
            if hasattr(loaded_model, "n_features_in_") and loaded_model.n_features_in_ != expected_dim:
                print(f"[WARNING] Saved model expects {loaded_model.n_features_in_} features, but current model uses {expected_dim} features.")
                print("[INFO] Outdated model file will be replaced after new calibration.")
                return False

            self.model = loaded_model
            self.is_trained = True
            print(f"[INFO] Gaze model loaded from {file_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load gaze model from {file_path}: {e}")
            return False
