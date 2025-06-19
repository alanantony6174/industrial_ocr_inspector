# File: ocr/camera.py

import cv2

# Try to import RealSense; if unavailable, we fall back to webcam only
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

class Webcam:
    def __init__(self, index=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def stop(self):
        self.cap.release()

class RealSense:
    def __init__(self, width=640, height=480, fps=30):
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 not installed or camera not connected")
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(cfg)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        c = frames.get_color_frame()
        if not c:
            return None
        import numpy as np
        return np.asanyarray(c.get_data())

    def stop(self):
        self.pipeline.stop()

def init_camera(kind="webcam", **kwargs):
    kind = kind.lower()
    if kind in ("realsense", "rs") and REALSENSE_AVAILABLE:
        return RealSense(**kwargs)
    else:
        return Webcam(**kwargs)

def get_frame(camera):
    return camera.get_frame()

def stop_camera(camera):
    camera.stop()
