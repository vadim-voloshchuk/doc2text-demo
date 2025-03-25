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
        "Ты — эксперт по анализу документов. На основе предоставленного текста определи тип документа и извлеки ключевую информацию. УЧТИ, что текст может быть искаженный, потому что вытаскивался из OCR  модели, попробуй определить смысл слов и восстановить их. "
        "Предоставь четко структурированный ответ в формате JSON со следующими полями: document_type (тип документа), title (название документа), "
        "author (автор документа), date (дата документа), summary (краткое описание), full_text (полный текст документа без сокращений и исправлений), "
        "keywords (ключевые слова для индексации и поиска). Текст документа:\n{}"
    ),
    "generate_specific_fields": (
        "Перечисли специфичные поля, которые обычно встречаются в документе указанного типа. Это поможет извлечь максимально полезную информацию. "
        "Предоставь список этих полей в виде JSON-массива. Тип документа: '{}'"
    ),
    "extract_detailed_fields": (
        "Ты — эксперт по детальному извлечению данных из документов. На основе указанного типа документа и полного текста, заполни следующие поля: "
        "document_type, title, author, date, summary, full_text, keywords, а также указанные специфичные поля: {}. "
        "Предоставь ответ строго в формате JSON. Текст документа:\n{}"
    ),
    "estimate_document_count": (
        "Оцени, сколько отдельных документов объединено в представленном тексте. "
        "Верни результат строго в формате JSON с полем 'document_count'. Текст документа:\n{}"
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
