import magic

def get_mime_type(file_obj):
    """
    Определяет MIME-тип файла.
    Если file_obj не является файловым объектом (например, строкой с путём),
    открывает файл в бинарном режиме и читает первые 1024 байта.
    """
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
        file_data = file_obj.read(1024)
        file_obj.seek(0)
    elif isinstance(file_obj, str):
        with open(file_obj, "rb") as f:
            file_data = f.read(1024)
    else:
        raise ValueError("Неподдерживаемый тип файла: {}".format(type(file_obj)))
    
    mime = magic.from_buffer(file_data, mime=True)
    return mime
