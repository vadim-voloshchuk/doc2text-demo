import os
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
from typing import List
from pdf2image import convert_from_path
import easyocr

# Инициализируем EasyOCR (русский + английский)
reader = easyocr.Reader(['ru', 'en'], gpu=False)

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Повышает контраст изображения с помощью CLAHE."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def denoise_image(image: np.ndarray) -> np.ndarray:
    """Убираем шум fastNlMeansDenoising."""
    return cv2.fastNlMeansDenoising(image, h=20)

def sharpen_image(image: np.ndarray) -> np.ndarray:
    """Повышаем резкость."""
    gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

def correct_perspective(image: np.ndarray) -> np.ndarray:
    """
    Ищет большой четырёхугольник и делает перспективное преобразование.
    Если не найден — возвращаем исходное изображение.
    """
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            break
    else:
        return image  # ничего не нашли

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA, widthB = np.linalg.norm(br - bl), np.linalg.norm(tr - tl)
    heightA, heightB = np.linalg.norm(tr - br), np.linalg.norm(tl - bl)
    maxWidth, maxHeight = int(max(widthA, widthB)), int(max(heightA, heightB))

    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

def binarize_image(image: np.ndarray) -> np.ndarray:
    """Адаптивная бинаризация (с проверкой среднего)."""
    if len(image.shape) == 2:
        gray = image
    else:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = image

    binarized = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=35, C=11
    )
    return binarized if np.mean(binarized) >= 50 else gray

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Полный pipeline: перспектива, контраст, шум, резкость, бинаризация, BGR.
    """
    image = correct_perspective(image)
    image = enhance_contrast(image)
    image = denoise_image(image)
    image = sharpen_image(image)
    image = binarize_image(image)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# -----------------------------------------------------------------------------
#                          Новый метод: bounding box из EasyOCR
# -----------------------------------------------------------------------------

def merge_overlapping_boxes(boxes, eps=50):
    """
    Сливаем пересекающиеся или близкие bounding boxes.
    eps = 50 пикселей — допустимое расстояние/зазор для объединения.
    boxes — список, где каждый элемент: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    Возвращаем список (minX, minY, maxX, maxY) блоков.
    """
    merged = []
    for box in boxes:
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        minx, maxx = min(x_coords), max(x_coords)
        miny, maxy = min(y_coords), max(y_coords)

        found_merge = False
        for i, (mx1, my1, mx2, my2) in enumerate(merged):
            # Проверка пересечения или близости
            # Если наши новые координаты не полностью за пределами уже имеющегося бокса
            if not (maxx < mx1 - eps or minx > mx2 + eps or maxy < my1 - eps or miny > my2 + eps):
                # Расширяем существующий бокс
                new_minx = min(minx, mx1)
                new_miny = min(miny, my1)
                new_maxx = max(maxx, mx2)
                new_maxy = max(maxy, my2)
                merged[i] = (new_minx, new_miny, new_maxx, new_maxy)
                found_merge = True
                break
        if not found_merge:
            merged.append((minx, miny, maxx, maxy))
    return merged

def split_image_by_ocr(image: np.ndarray) -> List[np.ndarray]:
    """
    1) Прогоняем EasyOCR (detail=1, paragraph=False),
    2) собираем bounding box'ы,
    3) сливаем их (merge_overlapping_boxes),
    4) вырезаем блоки.
    """
    # 1) EasyOCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray, detail=1, paragraph=False)
    if not results:
        return [image]

    # 2) Собираем bounding boxes
    boxes = []
    for item in results:
        if len(item) >= 2:  # (box, text[, conf])
            box = item[0]
            boxes.append(box)

    # 3) Сливаем пересекающиеся
    merged_boxes = merge_overlapping_boxes(boxes, eps=50)
    if not merged_boxes:
        return [image]

    # 4) Вырезаем каждый объединённый блок
    blocks = []
    for (minx, miny, maxx, maxy) in merged_boxes:
        # Учитываем границы
        minx, miny = max(0, int(minx)), max(0, int(miny))
        maxx, maxy = min(image.shape[1], int(maxx)), min(image.shape[0], int(maxy))
        w, h = maxx - minx, maxy - miny
        if w > 10 and h > 10:
            crop = image[miny:maxy, minx:maxx]
            blocks.append(crop)

    return blocks if blocks else [image]

# -----------------------------------------------------------------------------
#                          Обработка PDF / обычного изображения
# -----------------------------------------------------------------------------

def normalize_pdf(file_obj) -> List[str]:
    """
    Обрабатывает многостраничный PDF: для каждой страницы
    вызывает EasyOCR box'ы, preprocess и сохраняет.
    """
    paths = []
    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        # вместо file_obj.read() используем open(file_obj.name,'rb')
        with open(file_obj.name, 'rb') as f:
            pdf_data = f.read()
        tmp_pdf.write(pdf_data)
        tmp_pdf.flush()

        pages = convert_from_path(tmp_pdf.name, dpi=200)

    os.unlink(tmp_pdf.name)

    for i, page in enumerate(pages):
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        blocks = split_image_by_ocr(img)
        for j, region in enumerate(blocks):
            processed = preprocess_image(region)
            path = f"/tmp/doc2text_pdf_page_{i+1}_block_{j+1}.png"
            cv2.imwrite(path, processed)
            paths.append(path)
    return paths


def normalize_image(file_obj) -> List[str]:
    """
    Обычное изображение:
    1) Читаем,
    2) bounding box через OCR,
    3) Препроцессинг,
    4) Сохраняем
    """
    paths = []
    # Читаем с диска
    if hasattr(file_obj, 'name') and isinstance(file_obj.name, str):
        image = cv2.imread(file_obj.name)
    elif isinstance(file_obj, str):
        image = cv2.imread(file_obj)
    else:
        raise ValueError("Неподдерживаемый тип файла (нельзя прочитать из .name)") 

    blocks = split_image_by_ocr(image)
    for i, region in enumerate(blocks):
        processed = preprocess_image(region)
        path = f"/tmp/doc2text_img_block_{i+1}.png"
        cv2.imwrite(path, processed)
        paths.append(path)

    return paths

def normalize_file(file_obj) -> List[str]:
    """
    Определяет, PDF это или нет. Затем обрабатывает.
    """
    ext = os.path.splitext(file_obj.name)[-1].lower()
    if ext == '.pdf':
        return normalize_pdf(file_obj)
    return normalize_image(file_obj)
