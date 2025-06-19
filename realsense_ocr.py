import pyrealsense2 as rs
import cv2
import numpy as np
import json
from paddleocr import PPStructureV3

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# Initialize PaddleOCR
ocr_pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_formula_recognition=False,
    use_chart_recognition=False,
    chart_recognition_batch_size=1,
)

# Window setup
cv2.namedWindow("RealSense OCR", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RealSense OCR", 2560, 720)  # for side-by-side view

# Blank overlay image initially
overlay_img = np.zeros((720, 1280, 3), dtype=np.uint8)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())

    # Combine live and overlay side-by-side
    combined = np.hstack((frame, overlay_img))
    cv2.imshow("RealSense OCR", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC to exit
        break

    elif key == 32:  # SPACE to capture and OCR
        snapshot = frame.copy()
        snapshot_path = 'captured.png'
        cv2.imwrite(snapshot_path, snapshot)

        # Run OCR
        ocr_results = ocr_pipeline.predict(input=snapshot_path)

        # Save & load result JSON
        ocr_results[0].save_to_json("output")
        with open("output/captured_res.json") as f:
            data = json.load(f)

        # Draw overlays
        overlay_img = snapshot.copy()
        for poly, text in zip(data['overall_ocr_res']['rec_polys'], data['overall_ocr_res']['rec_texts']):
            pts = [(int(x), int(y)) for x, y in poly]
            cv2.polylines(overlay_img, [np.array(pts)], isClosed=True, color=(0,255,0), thickness=2)
            cv2.putText(overlay_img, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

pipeline.stop()
cv2.destroyAllWindows()
