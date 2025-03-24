from flask import Flask
from app.config import Config
from app.utils.logger import setup_logging

# Настройка логирования
setup_logging()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Импорт и регистрация маршрутов
    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    return app

app = create_app()
