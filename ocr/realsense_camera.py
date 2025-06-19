import pyrealsense2 as rs
import numpy as np

def init_camera(width=640, height=480, fps=30):
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(cfg)
    return pipeline

def get_frame(pipeline):
    frames = pipeline.wait_for_frames()
    c = frames.get_color_frame()
    if not c:
        return None
    return np.asanyarray(c.get_data())

def stop_camera(pipeline):
    pipeline.stop()
