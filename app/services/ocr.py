import logging
import os
from pdf2image import convert_from_path

from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from shiftlab_ocr.doc2text.reader import Reader

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Инициализация моделей
doctr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

def extract_text(file_obj, mime_type):
    """
    Извлекает текст из документа с использованием и shiftlab_ocr, и docTR.
    Приоритет отдаётся обоим, объединяя результат, если возможно.
    """

    shiftlab_reader = Reader()

    logger.info("Старт обработки файла. MIME-тип: %s", mime_type)

    # Подготовка временного пути
    temp_path = '/tmp/temp_document'
    try:
        ext = os.path.splitext(file_obj.name)[1] if hasattr(file_obj, 'name') else ''
    except Exception as e:
        logger.exception("Ошибка при определении расширения: %s", e)
        ext = ''
    if mime_type == "application/pdf":
        ext = ".pdf"
    elif not ext:
        ext = ".png"
    temp_path += ext

    # Чтение файла
    try:
        if hasattr(file_obj, "read"):
            data = file_obj.read()
        elif isinstance(file_obj, str):
            with open(file_obj, 'rb') as f:
                data = f.read()
        else:
            logger.error("Неподдерживаемый тип объекта file_obj: %s", type(file_obj))
            return None
    except Exception as e:
        logger.exception("Ошибка при чтении данных: %s", e)
        return None

    # Сохраняем файл
    try:
        with open(temp_path, 'wb') as f:
            f.write(data)
    except Exception as e:
        logger.exception("Ошибка при сохранении файла: %s", e)
        return None

    # Если PDF — конвертируем в PNG
    if mime_type == "application/pdf":
        try:
            pages = convert_from_path(temp_path, dpi=300)
            if pages:
                image = pages[0]
                temp_image_path = '/tmp/temp_document.png'
                image.save(temp_image_path, 'PNG')
                logger.info("PDF конвертирован в PNG.")
                temp_path = temp_image_path
            else:
                logger.error("Не удалось конвертировать PDF.")
                return None
        except Exception as e:
            logger.exception("Ошибка при конвертации PDF в PNG: %s", e)
            return None

    # === Распознавание через docTR ===
    doctr_text = None
    try:
        doc = DocumentFile.from_images(temp_path)
        result = doctr_model(doc)
        json_output = result.export()

        lines = []
        for page in json_output['pages']:
            for block in page['blocks']:
                for line in block['lines']:
                    text = " ".join([w['value'] for w in line['words']])
                    lines.append(text)

        if lines:
            doctr_text = "\n".join(lines)
            logger.info("docTR успешно извлёк текст.")
        else:
            logger.warning("docTR не извлёк текст.")
    except Exception as e:
        logger.exception("Ошибка docTR: %s", e)

    # === Распознавание через shiftlab_ocr ===
    shiftlab_text = None
    try:
        result = shiftlab_reader.doc2text(temp_path)
        if result and result[0].strip():
            shiftlab_text = result[0]
            logger.info("shiftlab_ocr успешно извлёк текст.")
        else:
            logger.warning("shiftlab_ocr не извлёк текст.")
    except Exception as e:
        logger.exception("Ошибка shiftlab_ocr: %s", e)

    # === Финальный результат ===
    if doctr_text and shiftlab_text:
        logger.info("Объединённый результат из двух OCR-движков.")
        return doctr_text + "\n\n---\n\n" + shiftlab_text
    elif doctr_text:
        logger.info("Возвращён результат только от docTR.")
        return doctr_text
    elif shiftlab_text:
        logger.info("Возвращён результат только от shiftlab_ocr.")
        return shiftlab_text
    else:
        logger.error("Ни один OCR-движок не извлёк текст.")
        return None
