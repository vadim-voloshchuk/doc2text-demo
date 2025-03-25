import os
import json
import re
import logging
from hugchat.login import Login
from hugchat.hugchat import ChatBot
from hugchat.exceptions import ChatError
from transliterate import translit

# Настройка логирования
logger = logging.getLogger("document_pipeline")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

OCR_FIX_MAP = {
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К",
    "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х",
    "y": "у", "6": "б", "0": "о", "3": "з", "4": "ч", "1": "л"
}

PROMPTS = {
    "analyze_text": (
        "Ты — эксперт по анализу документов. Проанализируй текст по шагам.\n\n"
        "# Шаг 1: Определи тип документа.\n"
        "Поясни, почему ты выбрал именно этот тип.\n\n"
        "# Шаг 2: Определи ключевые поля — название, автор, дата, краткое содержание, ключевые слова.\n"
        "Попробуй восстановить слова даже при ошибках OCR. Если поля не определяются — заполни как null.\n\n"
        "# Шаг 3: Сформируй итоговый JSON со следующими полями:\n"
        "- document_type\n- title\n- author\n- date\n- summary\n- keywords\n- full_text\n\n"
        "Ответ верни строго в формате:\n"
        "```json\n{{...}}\n```\n\n"
        "Вот текст документа:\n{}\n"
    ),

    "generate_specific_fields": (
        "Ты — специалист по структуре документов.\n"
        "На основе указанного типа, перечисли специфичные поля, часто встречающиеся в таких документах.\n"
        "Верни JSON-массив, например: [\"номер договора\", \"ИНН\", \"адрес\"]\n\n"
        "Тип документа: '{}'"
    ),

    "extract_detailed_fields": (
        "Ты — эксперт по извлечению полей из текста документов. Работай поэтапно:\n\n"
        "# Шаг 1: Прочитай полный текст и проанализируй общий смысл.\n"
        "# Шаг 2: Используя шаблон полей, попытайся извлечь значения. Учитывай ошибки OCR, исправляй и дополняй фразы по смыслу.\n"
        "# Шаг 3: Сформируй финальный JSON:\n"
        "- document_type\n- title\n- author\n- date\n- summary\n- keywords\n- full_text\n"
        "- и дополнительно поля: {{ {} }}\n\n"
        "Ответ строго в формате:\n"
        "```json\n{{...}}\n```\n\n"
        "Текст документа:\n{}"
    ),

    "estimate_document_count": (
        "Проанализируй, сколько отдельных документов содержится в тексте.\n"
        "Если видишь повторяющиеся шаблоны (например, даты + подписи), это может быть несколько документов.\n"
        "Ответ строго в JSON формате:\n"
        "{{ \"document_count\": N }}\n\n"
        "Текст:\n{}"
    )
}



def fix_ocr_translit(text):
    corrected = ''.join(OCR_FIX_MAP.get(ch, ch) for ch in text)
    if re.search(r'[a-zA-Z]', corrected):
        try:
            corrected = translit(corrected, 'ru')
        except Exception:
            pass
    return corrected

def authorize():
    EMAIL = os.environ.get('HF_EMAIL', 'your_email_here')
    PASSWD = os.environ.get('HF_PASS', 'your_password_here')
    cookie_path_dir = os.environ.get('HUGGING_CHAT_COOKIE_DIR', './cookies/')

    sign = Login(EMAIL, PASSWD)
    cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)
    chatbot = ChatBot(cookies=cookies.get_dict())
    return chatbot

def analyze_text(text):
    chatbot = authorize()
    MAX_TEXT_LENGTH = 2000
    truncated_text = text[:MAX_TEXT_LENGTH] if len(text) > MAX_TEXT_LENGTH else text
    # translated_text = fix_ocr_translit(truncated_text)

    prompt = PROMPTS["analyze_text"].format(truncated_text)
    try:
        response = chatbot.chat(prompt).wait_until_done()
        return json.loads(response)
    except:
        return {"markdown_response": response}

def generate_specific_fields(document_type):
    chatbot = authorize()
    prompt = PROMPTS["generate_specific_fields"].format(document_type)
    try:
        response = chatbot.chat(prompt).wait_until_done()
        return json.loads(response)
    except:
        return []

def extract_detailed_fields(full_text, document_type):
    chatbot = authorize()
    specific_fields = generate_specific_fields(document_type)
    translated_text = fix_ocr_translit(full_text)

    prompt = PROMPTS["extract_detailed_fields"].format(", ".join(specific_fields), translated_text)
    try:
        response = chatbot.chat(prompt).wait_until_done()
        return json.loads(response)
    except:
        return {"markdown_response": response}

def estimate_document_count(full_text):
    chatbot = authorize()
    translated_text = fix_ocr_translit(full_text)

    prompt = PROMPTS["estimate_document_count"].format(translated_text)
    try:
        response = chatbot.chat(prompt).wait_until_done()
        result = json.loads(response)
        return result.get("document_count", None)
    except:
        match = re.search(r'\d+', response)
        return int(match.group(0)) if match else None

def process_document_pipeline(ocr_text):
    base_result = analyze_text(ocr_text)
    document_type = base_result.get("document_type", "unknown")
    detailed_result = extract_detailed_fields(ocr_text, document_type)
    document_count = estimate_document_count(ocr_text)

    return {
        "document_count": document_count,
        "base_analysis": base_result,
        "detailed_analysis": detailed_result
    }
