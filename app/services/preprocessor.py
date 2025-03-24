from io import BytesIO
from PIL import Image

def normalize_image(file_obj):
    """
    Если нормализация не требуется, просто возвращает файловый объект.
    Если file_obj не поддерживает метод seek (например, это строка с путём),
    открывает файл в бинарном режиме и возвращает его.
    """
    if not hasattr(file_obj, "seek"):
        try:
            file_obj = open(file_obj, "rb")
        except Exception as e:
            print(f"Не удалось открыть файл: {e}")
            return file_obj
    file_obj.seek(0)
    return file_obj
