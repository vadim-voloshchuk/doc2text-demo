import argparse
from app.services import file_handler, preprocessor, ocr, analyzer

def process_file(file_path):
    # Открываем файл
    with open(file_path, 'rb') as f:
        # Определяем MIME-тип
        mime_type = file_handler.get_mime_type(f)

        # Если изображение, нормализуем
        if mime_type.startswith('image/'):
            f = preprocessor.normalize_image(f)

        # Извлекаем текст
        extracted_text = ocr.extract_text(f, mime_type)

        # Анализируем текст
        structured_data = analyzer.analyze_text(extracted_text)

        return structured_data

def main():
    parser = argparse.ArgumentParser(description='Конвертация и анализ документа.')
    parser.add_argument('--file', required=True, help='Путь к файлу документа')
    args = parser.parse_args()

    result = process_file(args.file)
    print(result)

if __name__ == '__main__':
    main()
