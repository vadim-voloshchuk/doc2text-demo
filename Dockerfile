FROM python:3.9-slim

WORKDIR /app

# Копируем файлы проекта
COPY . /app

# Устанавливаем зависимости
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Gradio использует порт 7860 по умолчанию
EXPOSE 7860

# Запускаем Gradio-демо
CMD ["python", "-m", "app.gradio_app"]
