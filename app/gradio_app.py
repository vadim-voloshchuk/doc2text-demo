import torch
import os
import json
import logging
import gradio as gr

# Патч для weights_only=False
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Разрешаем pickle загрузку YOLO-модели
from shiftlab_ocr.doc2text.yolov5.models.yolo import Model
torch.serialization.add_safe_globals([Model])

from app.services import file_handler, preprocessor, ocr, analyzer

# Логирование
logger = logging.getLogger("document_pipeline")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# Парсинг markdown_response и красивый вывод
def parse_analysis(result):
    base_md = result.get("base_analysis", {}).get("markdown_response", "")
    detail_md = result.get("detailed_analysis", {}).get("markdown_response", "")

    md_final = ""

    if "```json" in base_md:
        comment, json_part = base_md.split("```json")
        json_text = json_part.split("```")[0].strip()
        base_data = json.loads(json_text)

        md_final += f"### 🗒️ Комментарий базового анализа:\n\n{comment.strip()}\n\n"
        md_final += "### 📋 Основные поля документа:\n"
        md_final += (
            f"- **Тип документа:** {base_data.get('document_type', '—')}\n"
            f"- **Название:** {base_data.get('title', '—')}\n"
            f"- **Автор:** {base_data.get('author', '—')}\n"
            f"- **Дата:** {base_data.get('date', '—')}\n"
            f"- **Краткое описание:** {base_data.get('summary', '—')}\n"
            f"- **Ключевые слова:** {', '.join(base_data.get('keywords', []))}\n"
            f"- **Полный текст:** {base_data.get('full_text', '—')}\n"
        )
    else:
        md_final += base_md

    md_final += "\n---\n"
    md_final += "### 🔍 Детальный анализ документа:\n\n"
    md_final += detail_md

    return md_final

# Обработка документа
def process_document(file):
    if file is None:
        return "**Ошибка:** Файл не загружен."

    mime_type = file_handler.get_mime_type(file)

    if mime_type.startswith("image/"):
        file = preprocessor.normalize_image(file)

    extracted_text = ocr.extract_text(file, mime_type)
    if not extracted_text:
        return "**Ошибка:** Не удалось извлечь текст."

    result = analyzer.process_document_pipeline(extracted_text)
    formatted_result = parse_analysis(result)

    return formatted_result

# Gradio интерфейс с инструкцией, примерами, предпросмотром и прелоудером
with gr.Blocks() as demo:
    gr.Markdown("""
    # 📑 Doc2Text LLM Service

    Загрузите ваш документ (**PDF или изображение**).
    Обработка и анализ займут несколько минут. Пожалуйста, не закрывайте страницу.
    """)

    file_input = gr.File(label="📂 Загрузите документ")
    image_preview = gr.Image(label="🖼️ Предпросмотр документа")
    output_md = gr.Markdown(label="📝 Результаты анализа")

    submit_button = gr.Button("🔍 Анализировать")

    gr.Examples(
        examples=[
            "examples/e0bfc9da78d759e5174d70d32737d9c0.jpg",
            "examples/snils_-obrazec.jpg",
            "examples/vneshniy-vid-voditelskih-prav.jpg"
        ],
        inputs=file_input,
        label="🖼️ Примеры документов"
    )

    file_input.change(lambda file: file.name if file else None, inputs=file_input, outputs=image_preview)

    submit_button.click(
        fn=process_document,
        inputs=file_input,
        outputs=output_md,
        api_name="process_document"
    )

    submit_button.click(lambda: gr.update(interactive=False), None, submit_button)
    submit_button.click(lambda: gr.update(interactive=True), None, submit_button, queue=False)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=False
    )
