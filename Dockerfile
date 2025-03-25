FROM python:3.10

WORKDIR /app

# Устанавливаем системные зависимости для OpenCV, libmagic, poppler и других компонентов
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы проекта
COPY . /app

# Устанавливаем Python-зависимости
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 7860

CMD ["python", "-m", "app.gradio_app"]
