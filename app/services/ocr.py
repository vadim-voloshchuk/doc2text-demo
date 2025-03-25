import logging
import os
from difflib import SequenceMatcher

import easyocr
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from shiftlab_ocr.doc2text.reader import Reader

# Логирование
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Модели
doctr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', detect_language=True, pretrained=True)
easyocr_reader = easyocr.Reader(['ru', 'en'], gpu=False)


def visualize_ocr(lines, confidences, title="OCR"):
    html = f"<h4>{title}</h4><pre>"
    for line, conf in zip(lines, confidences):
        color = "#00cc44" if conf > 0.85 else "#ffaa00" if conf > 0.6 else "#ff3333"
        html += f"<span style='color:{color}'>{line}</span>\n"
    html += "</pre>"
    return html


def merge_ocr_results(texts):
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return None
    if len(texts) == 1:
        return texts[0]

    unique = []
    for t in texts:
        if all(SequenceMatcher(None, t, u).ratio() < 0.85 for u in unique):
            unique.append(t)

    best = max(unique, key=len)

    merged = best
    for other in unique:
        if other != best:
            merged += "\n\n--- OCR вариант ---\n\n" + other

    return merged


def extract_text_from_pages(file_obj):
    image_paths = file_obj

    full_texts = []
    ocr_details = {"docTR": "", "easyocr": "", "shiftlab": "", "visual": ""}

    for idx, img_path in enumerate(image_paths):
        logger.info("Обрабатываю страницу %d/%d", idx + 1, len(image_paths))

        doctr_text, doctr_lines, doctr_conf = None, [], []
        easy_text, easy_lines, easy_conf = None, [], []
        shiftlab_text = None

        # docTR
        try:
            doc = DocumentFile.from_images(img_path)
            result = doctr_model(doc)
            for page in result.export()['pages']:
                for block in page['blocks']:
                    for line in block['lines']:
                        text = " ".join([w['value'] for w in line['words']])
                        doctr_lines.append(text)
                        doctr_conf.append(min(len(text) / 80, 1.0))
            doctr_text = "\n".join(doctr_lines)
        except Exception as e:
            logger.exception("Ошибка docTR: %s", e)

        # EasyOCR
        try:
            results = easyocr_reader.readtext(img_path, detail=1)
            easy_lines = [txt for _, txt, _ in results]
            easy_conf = [conf for _, _, conf in results]
            easy_text = "\n".join(easy_lines)
        except Exception as e:
            logger.exception("Ошибка EasyOCR: %s", e)

        # Shiftlab OCR
        try:
            shiftlab_reader = Reader()
            result = shiftlab_reader.doc2text(img_path)
            shiftlab_text = result[0].strip() if result else ""
        except Exception as e:
            logger.exception("Ошибка Shiftlab OCR: %s", e)

        # Объединение
        final_page_text = merge_ocr_results([doctr_text, shiftlab_text, easy_text])
        if final_page_text:
            full_texts.append(final_page_text)

        # Визуализация
        visual_html = ""
        if doctr_lines:
            visual_html += visualize_ocr(doctr_lines, doctr_conf, "docTR")
        if easy_lines:
            visual_html += visualize_ocr(easy_lines, easy_conf, "EasyOCR")

        ocr_details["docTR"] += doctr_text or ""
        ocr_details["easyocr"] += easy_text or ""
        ocr_details["shiftlab"] += shiftlab_text or ""
        ocr_details["visual"] += visual_html

    final_document_text = " ".join(full_texts)
    logger.info("OCR обработка завершена")

    return final_document_text, ocr_details
