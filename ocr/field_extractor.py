# File: ocr/field_extractor.py

import json
import re
import math
from dateutil import parser

def load_ocr(path):
    with open(path, 'r') as f:
        return json.load(f)

def center_of(bbox):
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# date‐like regex (MM.YYYY, DD-MM-YYYY, MMM.YYYY, etc)
DATE_PAT = re.compile(
    r'(\d{1,2}[./-])?\d{1,2}[./-]\d{2,4}|'
    r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.?\s*\d{4}',
    re.IGNORECASE
)

def extract_fields(json_path, score_thresh=0.0):
    data = load_ocr(json_path)
    recs = data.get('overall_ocr_res', {})
    texts  = recs.get('rec_texts', [])
    boxes  = recs.get('rec_boxes', [])
    scores = recs.get('rec_scores', [])

    # build filtered blocks
    blocks = []
    for t, b, s in zip(texts, boxes, scores):
        if s < score_thresh:
            continue
        txt = t.strip()
        blocks.append({
            'text':   txt,
            'low':    txt.lower(),
            'bbox':   b,
            'center': center_of(b),
        })

    result = {'batch_no': None, 'mfg_date': None, 'exp_date': None}

    # 1) inline EXP / MFG / MFD
    for blk in blocks:
        txt, low = blk['text'], blk['low']

        # EXP
        if 'exp' in low:
            m = re.search(r'exp\.?\s*[:\-]?\s*(.+)', txt, re.IGNORECASE)
            result['exp_date'] = (m.group(1) if m else txt).strip()
            continue

        # MFG or MFD
        if 'mfg' in low or 'mfd' in low or 'manufacture' in low:
            m = re.search(r'(?:mfg|mfd|manufacture)\.?\s*[:\-]?\s*(.+)', txt, re.IGNORECASE)
            result['mfg_date'] = (m.group(1) if m else txt).strip()
            continue

    # 2) batch no
    batch_label = next(
        (b for b in blocks if re.search(r'\b(b\.?no|batch)\b', b['low'])),
        None
    )
    if batch_label:
        txt = batch_label['text']
        m = re.match(r'(?:b\.?no\.?)\s*[:\-]?\s*(.+)', txt, re.IGNORECASE)
        if m:
            result['batch_no'] = m.group(1).strip()
        else:
            c0 = batch_label['center']
            cands = [
                b for b in blocks
                if b is not batch_label
                and 'exp' not in b['low']
                and 'mfg' not in b['low']
                and 'mfd' not in b['low']
                and not DATE_PAT.search(b['text'])
            ]
            if cands:
                nearest = min(cands, key=lambda b: dist(c0, b['center']))
                result['batch_no'] = nearest['text']

    # 3) FALLBACK: last three lines if all still None
    if all(v is None for v in result.values()):
        if len(texts) >= 3:
            result['batch_no'], result['mfg_date'], result['exp_date'] = texts[-3:]

    # 4) ensure no None
    for k in result:
        if not result[k]:
            result[k] = 'N/A'

    return result
