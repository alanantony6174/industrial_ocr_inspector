import cv2
import json
import numpy as np

# Load original image
img = cv2.imread('test.png')

# Load OCR JSON
with open('output/test_res.json') as f:
    data = json.load(f)

# Draw all text boxes
for poly, text in zip(data['overall_ocr_res']['rec_polys'], data['overall_ocr_res']['rec_texts']):
    pts = [(int(x), int(y)) for x, y in poly]
    cv2.polylines(img, [np.array(pts)], isClosed=True, color=(0,255,0), thickness=2)
    cv2.putText(img, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

# Save visualization
cv2.imwrite('output/ocr_overlay.png', img)
