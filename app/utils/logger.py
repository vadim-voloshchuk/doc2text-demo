import logging

def setup_logging():
    """
    Настраивает базовое логирование для приложения.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )
