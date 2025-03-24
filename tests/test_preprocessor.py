import io
from PIL import Image
import pytest
from app.services import preprocessor

def test_normalize_image():
    # Создаем простое изображение в памяти
    image = Image.new('RGB', (100, 100), color='red')
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)

    normalized = preprocessor.normalize_image(buf)
    # Проверяем, что результат – BytesIO объект, содержащий изображение
    assert hasattr(normalized, 'read')
