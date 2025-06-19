from paddleocr import PPStructureV3

# Initialize once
ocr = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_formula_recognition=False,
    use_chart_recognition=False,
    chart_recognition_batch_size=1,
)

def run_ocr(image_path: str):
    """
    Runs PaddleOCR on the given image, saves JSON to output/logs/result.json,
    and returns the OCR result object.
    """
    result = ocr.predict(input=image_path)[0]
    result.save_to_json("output/logs/result.json")
    return result
