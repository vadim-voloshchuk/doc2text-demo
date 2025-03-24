import pytest
from app.services import analyzer

def test_analyze_text():
    sample_text = "Это пример документа. Автор: Иван Иванов, Дата: 2025-01-01."
    # Т.к. фактический вызов hugging-chat-api требует подключения, можно замокать функцию.
    # Здесь просто проверим, что возвращается значение (например, строка или словарь)
    result = analyzer.analyze_text(sample_text)
    # Допустим, ожидаем, что результат содержит ключ 'document_type'
    assert isinstance(result, dict) or 'document_type' in result
