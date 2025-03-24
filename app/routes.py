from flask import Blueprint, request, jsonify
from app.services import file_handler, preprocessor, ocr, analyzer

bp = Blueprint('main', __name__)

@bp.route('/extract-text', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400

    # Загрузка файла и определение MIME-типа
    file = request.files['file']
    mime_type = file_handler.get_mime_type(file)

    # Если это изображение, применяем нормализацию
    if mime_type.startswith('image/'):
        file = preprocessor.normalize_image(file)

    # Извлечение текста с использованием подходящего метода
    extracted_text = ocr.extract_text(file, mime_type)

    # Анализ текста и структурирование ответа через hugging-chat-api
    structured_data = analyzer.analyze_text(extracted_text)

    return jsonify(structured_data)
