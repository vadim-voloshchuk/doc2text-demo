import io
import pytest
from app.services import ocr

def test_extract_text_with_invalid_file():
    # Передаем несуществующий формат для теста
    dummy = io.BytesIO(b"Invalid content")
    dummy.name = "dummy.unknown"
    mime_type = "application/octet-stream"
    text = ocr.extract_text(dummy, mime_type)
    # Тестируем, что функция возвращает строку (в данном случае сообщение об ошибке)
    assert isinstance(text, str)
