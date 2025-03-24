FROM python:3.9-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends libmagic1 && \
    rm -rf /var/lib/apt/lists/*

# Копируем файлы проекта
COPY . /app

# Устанавливаем Python-зависимости
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 7860

CMD ["python", "-m", "app.gradio_app"]
