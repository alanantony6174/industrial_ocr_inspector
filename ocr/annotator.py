import cv2
import json
import numpy as np

def draw_overlay(image: np.ndarray, json_path: str):
    """
    Reads output/logs/result.json, draws polygons & texts over a copy of `image`,
    and returns (overlayed_image, list_of_texts).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)['overall_ocr_res']

    overlay = image.copy()
    texts = []
    for poly, text in zip(data['rec_polys'], data['rec_texts']):
        pts = np.array([[int(x), int(y)] for x, y in poly])
        cv2.polylines(overlay, [pts], isClosed=True, color=(0,255,0), thickness=2)
        cv2.putText(overlay, text, tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        texts.append(text)
    return overlay, texts
