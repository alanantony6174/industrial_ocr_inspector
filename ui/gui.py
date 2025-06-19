# File: ui/gui.py

import os, time, uuid
import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from threading import Thread

from ocr.realsense_camera import init_camera, get_frame, stop_camera
from ocr.ocr_engine       import run_ocr
from ocr.annotator        import draw_overlay

# Texture tags
LIVE_TEX    = "live_texture"
OVERLAY_TEX = "overlay_texture"

def bgr_to_rgba_norm(frame):
    """Convert BGR uint8 → flat list of normalized RGBA floats for DearPyGui."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    rgba = np.dstack((rgb, np.full((h, w), 255, dtype=np.uint8)))
    return (rgba.astype(np.float32) / 255.0).flatten().tolist()

def process_capture(frame):
    """Run OCR/annotator and update the GUI textures and text."""
    # 1) Save snapshot
    img_id = uuid.uuid4().hex
    img_path = f"output/images/{img_id}.png"
    os.makedirs("output/images", exist_ok=True)
    os.makedirs("output/logs", exist_ok=True)
    cv2.imwrite(img_path, frame)

    # 2) OCR + overlay
    run_ocr(img_path)
    ov_img, texts = draw_overlay(frame, "output/logs/result.json")

    # 3) Update overlay image
    dpg.set_value(OVERLAY_TEX, bgr_to_rgba_norm(ov_img))

    # 4) Update text list
    dpg.delete_item("text_list", children_only=True)
    for t in texts:
        dpg.add_text(t, parent="text_list")

def main_gui():
    dpg.create_context()

    # Start camera pipeline
    pipeline = init_camera()

    # Pre-create dynamic textures
    width, height = 640, 480
    dummy = [0.0] * (width * height * 4)
    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(width, height, dummy, tag=LIVE_TEX)
        dpg.add_dynamic_texture(width, height, dummy, tag=OVERLAY_TEX)

    # Build main window
    with dpg.window(label="Industrial OCR Inspector", width=1300, height=650):
        with dpg.group(horizontal=True):
            # Left: live camera feed
            with dpg.child_window(label="Live Feed", width=650, height=550):
                dpg.add_image(LIVE_TEX)

            # Right: overlay + text
            with dpg.group():
                with dpg.child_window(label="Captured Overlay", width=600, height=300):
                    dpg.add_image(OVERLAY_TEX)
                with dpg.child_window(label="OCR Text", width=600, height=230, tag="text_list"):
                    pass

        # Capture button below
        dpg.add_button(
            label="Capture OCR",
            width=200,
            height=30,
            callback=lambda: Thread(target=process_capture, args=(current_frame.copy(),)).start()
        )

    # Show viewport
    dpg.create_viewport(title="Industrial OCR", width=1320, height=700)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Main render loop
    global current_frame
    current_frame = np.zeros((height, width, 3), dtype=np.uint8)
    while dpg.is_dearpygui_running():
        frame = get_frame(pipeline)
        if frame is not None:
            current_frame = frame
            dpg.set_value(LIVE_TEX, bgr_to_rgba_norm(frame))
        dpg.render_dearpygui_frame()

    # Cleanup
    stop_camera(pipeline)
    dpg.destroy_context()

if __name__ == "__main__":
    main_gui()
