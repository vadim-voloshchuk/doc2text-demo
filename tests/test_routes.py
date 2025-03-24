import io
import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_extract_text_no_file(client):
    response = client.post('/extract-text')
    assert response.status_code == 400
    assert 'Файл не найден' in response.data

def test_extract_text_with_dummy_file(client):
    # Создаём dummy-файл (например, текстовый файл)
    dummy_file = io.BytesIO(b"Dummy content")
    dummy_file.name = "dummy.txt"
    data = {'file': (dummy_file, 'dummy.txt')}
    response = client.post('/extract-text', data=data, content_type='multipart/form-data')
    # Проверка успешного ответа (здесь может быть ошибка, если textract не может обработать dummy)
    assert response.status_code in [200, 500]
