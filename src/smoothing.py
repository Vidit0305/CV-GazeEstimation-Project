"""
Signal Smoothing and Noise Filtering Module.
Implements the One Euro (1€) Filter and Exponential Moving Average (EMA)
for real-time noise reduction without motion lag.
"""

import time
import math
import numpy as np
import config


class OneEuroFilter:
    """
    Adaptive Low-Pass Filter (One Euro Filter).
    Reduces high-frequency jitter when gaze is stationary while eliminating latency during fast movements.
    """

    def __init__(
        self,
        min_cutoff: float = config.ONE_EURO_MIN_CUTOFF,
        beta: float = config.ONE_EURO_BETA,
        d_cutoff: float = config.ONE_EURO_D_CUTOFF
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _smoothing_factor(dt: float, cutoff: float) -> float:
        """Calculates exponential smoothing coefficient alpha."""
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def filter(self, val: float, t: float | None = None) -> float:
        """Filters a scalar input value."""
        if t is None:
            t = time.time()

        if self.x_prev is None:
            self.x_prev = val
            self.dx_prev = 0.0
            self.t_prev = t
            return val

        dt = t - self.t_prev
        if dt <= 0.0:
            return self.x_prev

        # Estimate derivative (speed)
        dx = (val - self.x_prev) / dt
        a_d = self._smoothing_factor(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Adaptive cutoff frequency based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(dt, cutoff)

        # Filter value
        x_hat = a * val + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    def reset(self) -> None:
        """Resets filter memory state."""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class GazeSmoother:
    """Manages 2D gaze coordinate filtering for (x, y) coordinates."""

    def __init__(
        self,
        min_cutoff: float = config.ONE_EURO_MIN_CUTOFF,
        beta: float = config.ONE_EURO_BETA,
        d_cutoff: float = config.ONE_EURO_D_CUTOFF
    ):
        self.filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def update(self, gaze_x: float, gaze_y: float) -> tuple[float, float]:
        """Filters raw (gaze_x, gaze_y) position."""
        t = time.time()
        smooth_x = self.filter_x.filter(gaze_x, t)
        smooth_y = self.filter_y.filter(gaze_y, t)
        return smooth_x, smooth_y

    def reset(self) -> None:
        """Resets filter memory state."""
        self.filter_x.reset()
        self.filter_y.reset()
