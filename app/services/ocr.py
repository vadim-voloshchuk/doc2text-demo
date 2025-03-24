import logging
import os
from shiftlab_ocr.doc2text.reader import Reader
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def extract_text(file_obj, mime_type):
    """
    Извлекает текст из файла с использованием shiftlab_ocr.
    Если MIME-тип указывает на PDF, конвертирует первую страницу PDF в изображение.
    Если текст не извлечён (пустой результат), возвращает None.
    """
    logger.info("Начало извлечения текста с использованием shiftlab_ocr. MIME-тип: %s", mime_type)
    
    # Определяем временный путь и расширение
    temp_path = '/tmp/temp_document'
    try:
        if hasattr(file_obj, 'name'):
            ext = os.path.splitext(file_obj.name)[1]
        else:
            ext = ''
    except Exception as e:
        logger.exception("Ошибка при определении расширения: %s", e)
        ext = ''
    
    # Если PDF, установим расширение .pdf
    if mime_type == "application/pdf":
        ext = ".pdf"
    elif not ext:
        ext = ".png"
    temp_path += ext

    # Получаем данные из file_obj: если есть метод read, используем его, иначе предполагаем, что это путь
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

    # Сохраняем данные во временный файл
    try:
        with open(temp_path, 'wb') as f:
            f.write(data)
    except Exception as e:
        logger.exception("Ошибка при сохранении файла: %s", e)
        return None

    # Если PDF, конвертируем первую страницу в изображение
    if mime_type == "application/pdf":
        try:
            pages = convert_from_path(temp_path, dpi=300)
            if pages:
                image = pages[0]
                temp_image_path = '/tmp/temp_document.png'
                image.save(temp_image_path, 'PNG')
                logger.info("PDF успешно конвертирован в изображение.")
                temp_path = temp_image_path
            else:
                logger.error("Не удалось конвертировать PDF в изображение.")
                return None
        except Exception as e:
            logger.exception("Ошибка при конвертации PDF в изображение: %s", e)
            return None

    # Используем shiftlab_ocr для извлечения текста из изображения
    try:
        reader = Reader()
        result = reader.doc2text(temp_path)
        if result and result[0].strip():
            logger.info("shiftlab_ocr успешно извлек текст.")
            return result[0]
        else:
            logger.error("shiftlab_ocr не смог извлечь текст: пустой результат.")
            return None
    except Exception as e:
        logger.exception("Ошибка при выполнении shiftlab_ocr: %s", e)
        return None
