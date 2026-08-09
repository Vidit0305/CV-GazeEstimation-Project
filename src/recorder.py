"""
Screen and Camera Session Video Recorder Module.
Handles real-time desktop screen capture, gaze pointer & trail compositing,
live camera Picture-in-Picture (PiP) embedding, and asynchronous MP4 video writing.
"""

import os
import sys
import time
import datetime
import threading
import queue
import ctypes
import cv2
import numpy as np

import config

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False


class WindowsGDICapture:
    """Fallback Windows GDI screen capture using ctypes."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.user32 = ctypes.windll.user32
        self.gdi32 = ctypes.windll.gdi32

    def grab(self) -> np.ndarray | None:
        """Captures full screen frame via GDI BitBlt."""
        try:
            hdesktop = self.user32.GetDesktopWindow()
            desktop_dc = self.user32.GetDC(hdesktop)
            mem_dc = self.gdi32.CreateCompatibleDC(desktop_dc)
            hbitmap = self.gdi32.CreateCompatibleBitmap(desktop_dc, self.width, self.height)
            self.gdi32.SelectObject(mem_dc, hbitmap)

            # BitBlt entire screen (SRCCOPY = 0x00CC0020 | CAPTUREBLT = 0x40000000)
            self.gdi32.BitBlt(mem_dc, 0, 0, self.width, self.height, desktop_dc, 0, 0, 0x00CC0020 | 0x40000000)

            # Extract bitmap bits
            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ('biSize', ctypes.c_uint32),
                    ('biWidth', ctypes.c_int32),
                    ('biHeight', ctypes.c_int32),
                    ('biPlanes', ctypes.c_uint16),
                    ('biBitCount', ctypes.c_uint16),
                    ('biCompression', ctypes.c_uint32),
                    ('biSizeImage', ctypes.c_uint32),
                    ('biXPelsPerMeter', ctypes.c_int32),
                    ('biYPelsPerMeter', ctypes.c_int32),
                    ('biClrUsed', ctypes.c_uint32),
                    ('biClrImportant', ctypes.c_uint32)
                ]

            bmi = BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.biWidth = self.width
            bmi.biHeight = -self.height  # Top-down DIB
            bmi.biPlanes = 1
            bmi.biBitCount = 32
            bmi.biCompression = 0

            buffer = ctypes.create_string_buffer(self.width * self.height * 4)
            self.gdi32.GetDIBits(
                mem_dc,
                hbitmap,
                0,
                self.height,
                buffer,
                ctypes.byref(bmi),
                0
            )

            # Cleanup GDI objects
            self.gdi32.DeleteObject(hbitmap)
            self.gdi32.DeleteDC(mem_dc)
            self.user32.ReleaseDC(hdesktop, desktop_dc)

            # Convert BGRA to BGR
            img = np.frombuffer(buffer, dtype=np.uint8).reshape((self.height, self.width, 4))
            return img[:, :, :3].copy()
        except Exception:
            return None


class ScreenSessionRecorder:
    """
    Asynchronous Screen & Camera Session Video Recorder.
    Captures live desktop content, overlays real-time eye gaze point and trail,
    embeds live webcam Picture-in-Picture (PiP), and writes high-definition MP4.
    """

    def __init__(self):
        self.is_recording_active = False
        self.video_writer = None
        self.output_filepath = None
        self.start_time = 0.0
        self.frame_count = 0
        self.screen_w = 1920
        self.screen_h = 1080

        # Screen capture backend
        self.sct = mss.mss() if HAS_MSS else None
        self.gdi_fallback = None

        # Asynchronous frame queue & worker thread
        self.frame_queue = queue.Queue(maxsize=180)
        self.worker_thread = None
        self.stop_signal = threading.Event()

        # Gaze trail history for smooth motion rendering in video
        self.gaze_trail = []
        self.max_trail = config.GAZE_TRAIL_LENGTH

        # PiP Configuration
        self.pip_width = config.PIP_WIDTH
        self.pip_height = config.PIP_HEIGHT
        self.pip_position = config.PIP_POSITION
        self.pip_padding = config.PIP_PADDING
        self.include_camera = config.INCLUDE_CAMERA_IN_RECORDING
        self.include_trail = config.INCLUDE_GAZE_TRAIL_IN_RECORDING

    def _get_capture_monitor_rect(self) -> dict:
        """Returns monitor rect for mss capture."""
        return {
            "top": 0,
            "left": 0,
            "width": self.screen_w,
            "height": self.screen_h
        }

    def grab_desktop_frame(self) -> np.ndarray | None:
        """Captures the live desktop screen as a BGR numpy image."""
        if HAS_MSS and self.sct is not None:
            try:
                # Capture primary monitor (monitors[1] is primary, monitors[0] is all combined)
                mon = self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]
                screenshot = self.sct.grab(mon)
                # mss returns BGRA array
                frame_bgra = np.array(screenshot, dtype=np.uint8)
                return frame_bgra[:, :, :3]
            except Exception:
                pass

        # GDI Fallback on Windows
        if sys.platform == "win32":
            if self.gdi_fallback is None:
                self.gdi_fallback = WindowsGDICapture(self.screen_w, self.screen_h)
            frame = self.gdi_fallback.grab()
            if frame is not None:
                return frame

        # Fallback empty canvas if screenshot fails
        return np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

    def start_recording(self, screen_w: int, screen_h: int) -> str:
        """
        Starts a new video recording session.
        Returns the output file path.
        """
        if self.is_recording_active:
            return self.output_filepath

        self.screen_w = screen_w
        self.screen_h = screen_h
        self.gaze_trail.clear()
        self.frame_count = 0
        self.start_time = time.time()

        # Generate timestamped filename
        now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"gaze_session_{now_str}.mp4"
        self.output_filepath = os.path.join(config.RECORDINGS_DIR, filename)

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*config.RECORDING_FOURCC)
        self.video_writer = cv2.VideoWriter(
            self.output_filepath,
            fourcc,
            config.RECORDING_FPS,
            (self.screen_w, self.screen_h)
        )

        # Fallback to AVI if MP4 codec fails on system
        if not self.video_writer.isOpened():
            filename_avi = f"gaze_session_{now_str}.avi"
            self.output_filepath = os.path.join(config.RECORDINGS_DIR, filename_avi)
            fourcc_avi = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(
                self.output_filepath,
                fourcc_avi,
                config.RECORDING_FPS,
                (self.screen_w, self.screen_h)
            )

        if not self.video_writer.isOpened():
            print(f"[ERROR] Failed to open VideoWriter for {self.output_filepath}")
            self.is_recording_active = False
            return ""

        # Launch Asynchronous Writer Thread
        self.stop_signal.clear()
        # Empty queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.worker_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.worker_thread.start()

        self.is_recording_active = True
        print(f"\n[RECORDING STARTED] Session recording to: {self.output_filepath}")
        print(f"   Resolution: {self.screen_w}x{self.screen_h} | Target FPS: {config.RECORDING_FPS}")
        return self.output_filepath

    def _writer_worker(self) -> None:
        """Background worker thread that consumes frames from queue and writes to disk."""
        while not self.stop_signal.is_set() or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if self.video_writer is not None and frame is not None:
                    self.video_writer.write(frame)
                    self.frame_count += 1
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[WARNING] Video writer worker error: {e}")

    def process_and_record_frame(
        self,
        gaze_px: tuple[int, int],
        tracking_active: bool,
        camera_frame: np.ndarray | None = None,
        confidence: float = 0.0,
        direction_text: str = "CENTER",
        fps: float = 30.0
    ) -> None:
        """
        Grabs desktop screen, overlays gaze pointer, trails, camera PiP, and recording HUD,
        then queues the composited frame for asynchronous encoding.
        """
        if not self.is_recording_active:
            return

        # 1. Grab Live Desktop Screen
        screen_frame = self.grab_desktop_frame()
        if screen_frame is None:
            screen_frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        elif screen_frame.shape[0] != self.screen_h or screen_frame.shape[1] != self.screen_w:
            screen_frame = cv2.resize(screen_frame, (self.screen_w, self.screen_h), interpolation=cv2.INTER_AREA)

        # 2. Update Gaze Trail
        if tracking_active:
            self.gaze_trail.append(gaze_px)
            if len(self.gaze_trail) > self.max_trail:
                self.gaze_trail.pop(0)
        else:
            if self.gaze_trail:
                self.gaze_trail.pop(0)

        # 3. Render Smooth Gaze Trail on Desktop
        if self.include_trail and len(self.gaze_trail) > 1:
            for i in range(len(self.gaze_trail) - 1):
                pt1 = self.gaze_trail[i]
                pt2 = self.gaze_trail[i + 1]
                alpha = (i + 1) / float(len(self.gaze_trail))
                thickness = int(2 + 3 * alpha)
                trail_color = (
                    int(255 * (1 - alpha)),
                    int(220 * alpha),
                    int(255 * alpha)
                )
                cv2.line(screen_frame, pt1, pt2, trail_color, thickness, cv2.LINE_AA)

        # 4. Render Glowing Gaze Target Indicator
        if tracking_active:
            gx, gy = gaze_px
            gx = int(np.clip(gx, 20, self.screen_w - 20))
            gy = int(np.clip(gy, 20, self.screen_h - 20))

            # Outer glowing halo
            overlay = screen_frame.copy()
            cv2.circle(overlay, (gx, gy), config.DOT_RADIUS + 8, (255, 230, 0), -1)  # BGR Cyan/Yellow glow
            cv2.addWeighted(overlay, 0.35, screen_frame, 0.65, 0, screen_frame)

            # Main Cyan Circle
            cv2.circle(screen_frame, (gx, gy), config.DOT_RADIUS, (255, 230, 0), -1, cv2.LINE_AA)
            cv2.circle(screen_frame, (gx, gy), config.DOT_RADIUS, (255, 255, 255), 2, cv2.LINE_AA)

            # Center Bullseye Dot
            cv2.circle(screen_frame, (gx, gy), 4, (0, 0, 255), -1, cv2.LINE_AA)

            # Crosshairs
            line_color = (255, 255, 255)
            cv2.line(screen_frame, (gx - 22, gy), (gx - 10, gy), line_color, 2, cv2.LINE_AA)
            cv2.line(screen_frame, (gx + 10, gy), (gx + 22, gy), line_color, 2, cv2.LINE_AA)
            cv2.line(screen_frame, (gx, gy - 22), (gx, gy - 10), line_color, 2, cv2.LINE_AA)
            cv2.line(screen_frame, (gx, gy + 10), (gx, gy + 22), line_color, 2, cv2.LINE_AA)

        # 5. Embed Live Camera Picture-in-Picture (PiP)
        if self.include_camera and camera_frame is not None:
            self._embed_camera_pip(screen_frame, camera_frame, tracking_active, confidence)

        # 6. Render Recording Status Banner (Top-Left Badge)
        self._render_recording_banner(screen_frame, fps)

        # 7. Queue frame for writing without blocking main loop
        try:
            self.frame_queue.put_nowait(screen_frame)
        except queue.Full:
            # Drop frame if disk queue is overloaded to prevent memory leak
            pass

    def _embed_camera_pip(
        self,
        screen_frame: np.ndarray,
        camera_frame: np.ndarray,
        tracking_active: bool,
        confidence: float
    ) -> None:
        """Draws camera frame in corner with stylish card frame."""
        h, w, _ = screen_frame.shape
        pw, ph = self.pip_width, self.pip_height
        pad = self.pip_padding

        # Calculate PiP bounds
        if self.pip_position == "bottom_right":
            x1, y1 = w - pw - pad, h - ph - pad
        elif self.pip_position == "top_right":
            x1, y1 = w - pw - pad, pad + 40
        elif self.pip_position == "bottom_left":
            x1, y1 = pad, h - ph - pad
        else:  # top_left
            x1, y1 = pad, pad + 40

        x2, y2 = x1 + pw, y1 + ph

        # Ensure inside screen bounds
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return

        # Resize camera feed
        resized_cam = cv2.resize(camera_frame, (pw, ph), interpolation=cv2.INTER_AREA)

        # Card shadow/backdrop
        backdrop = screen_frame.copy()
        cv2.rectangle(backdrop, (x1 - 4, y1 - 24), (x2 + 4, y2 + 4), (12, 16, 24), -1)
        cv2.addWeighted(backdrop, 0.85, screen_frame, 0.15, 0, screen_frame)

        # Place camera feed
        screen_frame[y1:y2, x1:x2] = resized_cam

        # Border
        border_color = (0, 230, 140) if tracking_active else (80, 80, 240)
        cv2.rectangle(screen_frame, (x1, y1), (x2, y2), border_color, 2)
        cv2.rectangle(screen_frame, (x1 - 4, y1 - 24), (x2 + 4, y2 + 4), (60, 80, 110), 1)

        # PiP Header Tag
        cv2.putText(
            screen_frame,
            "AI LIVE WEBCAM",
            (x1 + 6, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 230, 255),
            1,
            cv2.LINE_AA
        )

        # Status badge in PiP
        status_text = f"TRACKING: {confidence:.0f}%" if tracking_active else "FACE LOST"
        status_color = (0, 255, 150) if tracking_active else (100, 100, 255)
        cv2.putText(
            screen_frame,
            status_text,
            (x2 - 110, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            status_color,
            1,
            cv2.LINE_AA
        )

    def _render_recording_banner(self, screen_frame: np.ndarray, fps: float) -> None:
        """Draws recording badge at top left."""
        elapsed_sec = int(time.time() - self.start_time)
        mins = elapsed_sec // 60
        secs = elapsed_sec % 60
        time_str = f"{mins:02d}:{secs:02d}"

        # Pulsing Red Dot (cycles every second)
        pulse = (int(time.time() * 2) % 2) == 0
        dot_color = (40, 40, 255) if pulse else (20, 20, 180)

        # Badge background
        bx, by, bw, bh = 20, 20, 210, 34
        overlay = screen_frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (15, 18, 26), -1)
        cv2.addWeighted(overlay, 0.80, screen_frame, 0.20, 0, screen_frame)
        cv2.rectangle(screen_frame, (bx, by), (bx + bw, by + bh), (60, 75, 100), 1)

        # Draw Red Recording Dot
        cv2.circle(screen_frame, (bx + 16, by + bh // 2), 6, dot_color, -1, cv2.LINE_AA)
        cv2.circle(screen_frame, (bx + 16, by + bh // 2), 7, (255, 255, 255), 1, cv2.LINE_AA)

        # REC Text & Timer
        rec_text = f"REC  {time_str}  |  {fps:.0f} FPS"
        cv2.putText(
            screen_frame,
            rec_text,
            (bx + 32, by + bh // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 245, 255),
            1,
            cv2.LINE_AA
        )

    def stop_recording(self) -> dict | None:
        """
        Stops the recording session and releases video resources.
        Returns summary statistics dictionary.
        """
        if not self.is_recording_active:
            return None

        self.is_recording_active = False
        duration_sec = max(0.1, time.time() - self.start_time)

        print("\n[INFO] Finalizing video file... please wait...")

        # Signal worker thread to finish remaining queue
        self.stop_signal.set()
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)

        # Release OpenCV VideoWriter
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        # Check output file size
        file_size_mb = 0.0
        if self.output_filepath and os.path.exists(self.output_filepath):
            file_size_mb = os.path.getsize(self.output_filepath) / (1024 * 1024)

        avg_fps = self.frame_count / duration_sec if duration_sec > 0 else 0.0

        stats = {
            "filepath": self.output_filepath,
            "duration_sec": duration_sec,
            "duration_str": f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}",
            "frames": self.frame_count,
            "avg_fps": avg_fps,
            "size_mb": file_size_mb
        }

        print("=" * 60)
        print(" [RECORDING COMPLETE & SAVED]")
        print(f" File Saved: {stats['filepath']}")
        print(f" Duration:   {stats['duration_str']} ({stats['duration_sec']:.1f}s)")
        print(f" Frames:     {stats['frames']} frames (avg {stats['avg_fps']:.1f} FPS)")
        print(f" File Size:  {stats['size_mb']:.2f} MB")
        print("=" * 60 + "\n")

        return stats

    def toggle_recording(self, screen_w: int, screen_h: int) -> bool:
        """Toggles recording on/off. Returns True if now recording, False if stopped."""
        if self.is_recording_active:
            self.stop_recording()
            return False
        else:
            self.start_recording(screen_w, screen_h)
            return True

    def is_recording(self) -> bool:
        """Returns True if currently recording video."""
        return self.is_recording_active

    def get_duration_str(self) -> str:
        """Returns formatted 'MM:SS' of current recording duration."""
        if not self.is_recording_active:
            return "00:00"
        elapsed = int(time.time() - self.start_time)
        return f"{elapsed // 60:02d}:{elapsed % 60:02d}"
