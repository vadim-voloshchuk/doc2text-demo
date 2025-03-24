import os
import json
import re
import logging
from hugchat.login import Login
from hugchat.hugchat import ChatBot
from hugchat.exceptions import ChatError

# Настройка логирования
logger = logging.getLogger("document_pipeline")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

def analyze_text(text):
    """
    Отправляет извлечённый текст в hugging-chat-api для базового анализа и структурирования.
    Возвращает JSON с полями: document_type, title, author, date, summary, full_text, keywords.
    Если разобрать JSON не получается, возвращает ответ в виде markdown_response.
    """
    logger.info("Запуск базового анализа текста.")
    EMAIL = os.environ.get('HF_EMAIL', 'your_email_here')
    PASSWD = os.environ.get('HF_PASS', 'your_password_here')
    cookie_path_dir = os.environ.get('HUGGING_CHAT_COOKIE_DIR', './cookies/')
    
    try:
        sign = Login(EMAIL, PASSWD)
        cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)
        chatbot = ChatBot(cookies=cookies.get_dict())
        logger.info("Авторизация для базового анализа успешна.")
    except Exception as e:
        logger.exception("Ошибка авторизации для базового анализа: %s", e)
        return {"error": "Ошибка авторизации для базового анализа."}
    
    MAX_TEXT_LENGTH = 2000
    if len(text) > MAX_TEXT_LENGTH:
        truncated_text = text[:MAX_TEXT_LENGTH] + "\n\n[Текст обрезан из-за ограничений]"
        logger.info("Текст обрезан до %d символов.", MAX_TEXT_LENGTH)
    else:
        truncated_text = text

    prompt = (
        "Проанализируй следующий текст документа и определи его тип. "
        "Сформируй JSON с полями: document_type, title, author, date, summary, full_text, keywords.\n\n"
        f"Текст документа:\n{truncated_text}"
    )
    logger.debug("Prompt для базового анализа сформирован.")
    
    try:
        response = chatbot.chat(prompt).wait_until_done()
        logger.info("Получен ответ базового анализа.")
    except ChatError as e:
        logger.exception("Ошибка при базовом анализе: %s", e)
        return {"markdown_response": str(e)}

    try:
        result = json.loads(response)
        logger.info("Базовый анализ успешно разобран как JSON.")
        return result
    except Exception as e:
        logger.exception("Не удалось разобрать JSON базового анализа: %s", e)
        return {"markdown_response": response}

def generate_specific_fields(document_type):
    """
    На основе типа документа генерирует список специфичных полей.
    Например, для "счет-фактуры" возвращает JSON-массив с типичными полями.
    Если не удаётся разобрать ответ, возвращает пустой список.
    """
    logger.info("Генерация специфичных полей для документа типа '%s'.", document_type)
    EMAIL = os.environ.get('HF_EMAIL', 'your_email_here')
    PASSWD = os.environ.get('HF_PASS', 'your_password_here')
    cookie_path_dir = os.environ.get('HUGGING_CHAT_COOKIE_DIR', './cookies/')
    
    try:
        sign = Login(EMAIL, PASSWD)
        cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)
        chatbot = ChatBot(cookies=cookies.get_dict())
        logger.info("Авторизация для генерации специфичных полей успешна.")
    except Exception as e:
        logger.exception("Ошибка авторизации для генерации специфичных полей: %s", e)
        return []
    
    prompt = (
        f"Для документа типа '{document_type}' перечисли типичные специфичные поля, которые следует извлечь. "
        "Верни ответ в виде JSON массива, например: [\"invoice_number\", \"vendor\", \"total_amount\", \"date_of_issue\"]"
    )
    logger.debug("Prompt для генерации специфичных полей сформирован.")
    
    try:
        response = chatbot.chat(prompt).wait_until_done()
        fields = json.loads(response)
        logger.info("Специфичные поля успешно сгенерированы: %s", fields)
        return fields
    except Exception as e:
        logger.exception("Ошибка при генерации специфичных полей: %s", e)
        return []

def extract_detailed_fields(full_text, document_type):
    """
    Выполняет детальный анализ полного текста с учетом специфичных полей,
    сгенерированных на основе типа документа.
    Если JSON не может быть разобран, возвращает ответ в виде markdown_response.
    """
    logger.info("Запуск детального анализа для документа типа '%s'.", document_type)
    specific_fields = generate_specific_fields(document_type)
    
    EMAIL = os.environ.get('HF_EMAIL', 'your_email_here')
    PASSWD = os.environ.get('HF_PASS', 'your_password_here')
    cookie_path_dir = os.environ.get('HUGGING_CHAT_COOKIE_DIR', './cookies/')
    
    try:
        sign = Login(EMAIL, PASSWD)
        cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)
        chatbot = ChatBot(cookies=cookies.get_dict())
        logger.info("Авторизация для детального анализа успешна.")
    except Exception as e:
        logger.exception("Ошибка авторизации для детального анализа: %s", e)
        return {"error": "Ошибка авторизации для детального анализа."}
    
    prompt = (
        f"Документ имеет тип: {document_type}.\n"
        "На основании полного текста, приведённого ниже, извлеки и заполни следующие поля:\n"
        "  - document_type\n"
        "  - title\n"
        "  - author\n"
        "  - date\n"
        "  - summary\n"
        "  - full_text\n"
        "  - keywords\n"
        f"Дополнительно извлеки следующие специфичные поля: {', '.join(specific_fields)}\n\n"
        "Текст документа:\n" + full_text
    )
    logger.debug("Prompt для детального анализа сформирован.")
    
    try:
        response = chatbot.chat(prompt).wait_until_done()
        logger.info("Получен ответ детального анализа.")
    except ChatError as e:
        logger.exception("Ошибка детального анализа: %s", e)
        return {"markdown_response": str(e)}
    
    try:
        result = json.loads(response)
        logger.info("Детальный анализ успешно разобран как JSON.")
        return result
    except Exception as e:
        logger.exception("Ошибка разбора JSON детального анализа: %s", e)
        return {"markdown_response": response}

def estimate_document_count(full_text):
    """
    Оценивает, сколько отдельных документов содержится в предоставленном тексте.
    Возвращает число или None, если не удалось определить.
    """
    logger.info("Запуск оценки количества документов в тексте.")
    EMAIL = os.environ.get('HF_EMAIL', 'your_email_here')
    PASSWD = os.environ.get('HF_PASS', 'your_password_here')
    cookie_path_dir = os.environ.get('HUGGING_CHAT_COOKIE_DIR', './cookies/')
    
    try:
        sign = Login(EMAIL, PASSWD)
        cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)
        chatbot = ChatBot(cookies=cookies.get_dict())
        logger.info("Авторизация для оценки количества документов успешна.")
    except Exception as e:
        logger.exception("Ошибка авторизации для оценки количества документов: %s", e)
        return None
    
    prompt = (
        "На основе следующего текста, пожалуйста, оцени, сколько отдельных документов присутствует на скане. "
        "Верни ответ в формате JSON, например: {\"document_count\": 2}\n\n"
        "Текст документа:\n" + full_text
    )
    logger.debug("Prompt для оценки количества документов сформирован.")
    
    try:
        response = chatbot.chat(prompt).wait_until_done()
        logger.info("Получен ответ оценки количества документов.")
    except ChatError as e:
        logger.exception("Ошибка при оценке количества документов: %s", e)
        return None

    try:
        result = json.loads(response)
        document_count = result.get("document_count", None)
        logger.info("Оценка количества документов: %s", document_count)
        return document_count
    except Exception as e:
        logger.exception("Ошибка разбора JSON оценки количества документов: %s", e)
        match = re.search(r'\d+', response)
        if match:
            document_count = int(match.group(0))
            logger.info("Извлечено число документов из строки: %d", document_count)
            return document_count
        return None

def process_document_pipeline(ocr_text):
    """
    Объединяет базовый анализ, детальный анализ и оценку количества документов.
    """
    logger.info("Запуск полного процесса анализа документа.")
    base_result = analyze_text(ocr_text)
    document_type = base_result.get("document_type", "unknown")
    detailed_result = extract_detailed_fields(ocr_text, document_type)
    document_count = estimate_document_count(ocr_text)
    
    combined = {
        "document_count": document_count,
        "base_analysis": base_result,
        "detailed_analysis": detailed_result
    }
    logger.info("Полный процесс анализа завершён.")
    return combined
