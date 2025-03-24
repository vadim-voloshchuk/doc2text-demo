import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default_secret_key')
    # Параметры для интеграции с hugging-chat-api
    HUGGING_CHAT_API_KEY = os.environ.get('HUGGING_CHAT_API_KEY', 'your_api_key_here')
    # Дополнительные настройки
    DEBUG = True
