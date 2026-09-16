"""
Pipeline de procesamiento de documentos.

Convierte una fotografía de un documento (licencia, SSN card, comprobante
de domicilio) tomada con celular en una versión "tipo escáner": recortada
al borde real del documento, con perspectiva corregida, enderezada, con
mejor iluminación y contraste.

IMPORTANTE: nada aquí es generativo. Todas las transformaciones son
deterministas (crop, perspective warp, CLAHE, sharpening) — nunca se
inventan ni reconstruyen datos del documento.
"""

import io
import numpy as np
import cv2
from PIL import Image, ExifTags


# ---------------------------------------------------------------------
# 1. Normalizar orientación EXIF (fotos de celular a veces vienen "giradas"
#    solo por metadata, no por los píxeles reales)
# ---------------------------------------------------------------------
def load_image_with_exif_correction(image_bytes: bytes) -> np.ndarray:
    pil_img = Image.open(io.BytesIO(image_bytes))
    try:
        for orientation_tag in ExifTags.TAGS:
            if ExifTags.TAGS[orientation_tag] == "Orientation":
                break
        exif = pil_img._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation_tag)
            rotations = {3: 180, 6: 270, 8: 90}
            if orientation_value in rotations:
                pil_img = pil_img.rotate(rotations[orientation_value], expand=True)
    except (AttributeError, KeyError, TypeError):
        pass  # sin EXIF o sin tag de orientación, seguimos sin rotar

    pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------
# 2-3. Detección de bordes del documento + esquinas
# ---------------------------------------------------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_document_corners(image: np.ndarray):
    """
    Intenta encontrar las 4 esquinas del documento en la foto.
    Devuelve (corners, confidence) donde corners es un array 4x2 en las
    coordenadas de la imagen ORIGINAL, o (None, 0) si no hay confianza
    suficiente (en ese caso el frontend debe pedir corrección manual).
    """
    h, w = image.shape[:2]
    scale = 1000.0 / max(h, w)
    small = cv2.resize(image, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    image_area = small.shape[0] * small.shape[1]
    best_quad = None
    best_score = 0.0

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            area_fraction = area / image_area
            if 0.05 <= area_fraction <= 0.97:
                rect = cv2.minAreaRect(c)
                (_, _), (rw, rh), _ = rect
                rectangularity = area / (rw * rh) if rw * rh > 0 else 0
                # La confianza se basa en qué tan limpio/rectangular es el
                # contorno encontrado, NO en qué tan grande es dentro del
                # encuadre — un documento fotografiado con mucha mesa
                # alrededor es tan válido como uno que llena la foto.
                score = rectangularity * 100
                if score > best_score:
                    best_score = score
                    best_quad = approx.reshape(4, 2).astype("float32")

    if best_quad is None:
        return None, 0.0

    # Reescalar las esquinas de vuelta a la resolución original
    corners = order_points(best_quad) / scale
    confidence = min(best_score, 99.0)
    return corners, confidence


# ---------------------------------------------------------------------
# 4-7. Perspective transform, deskew, iluminación, contraste, nitidez
# ---------------------------------------------------------------------
def warp_and_enhance(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    (tl, tr, br, bl) = corners

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(corners.astype("float32"), dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return enhance_image(warped)


def enhance_image(image: np.ndarray) -> np.ndarray:
    """Iluminación/contraste (CLAHE) + nitidez ligera. No generativo."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Unsharp mask suave — mejora legibilidad sin verse artificial
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(enhanced, 1.4, gaussian, -0.4, 0)

    return sharpened


# ---------------------------------------------------------------------
# 8. Validación de calidad
# ---------------------------------------------------------------------
def quality_checks(image: np.ndarray) -> list[str]:
    flags = []
    h, w = image.shape[:2]

    if min(h, w) < 400:
        flags.append("LOW_RESOLUTION")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 60:
        flags.append("POSSIBLY_BLURRY")

    # Glare real: una MANCHA compacta y aislada de blanco puro (reflejo de
    # flash), distinta del fondo blanco general del papel del documento
    # (que suele ser una sola región grande conectada, no una mancha chica).
    _, saturated = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(saturated, connectivity=8)
    image_area = h * w
    for label in range(1, num_labels):  # 0 es el fondo (no-blanco)
        area = stats[label, cv2.CC_STAT_AREA]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]
        bbox_area = bw * bh
        extent = area / bbox_area if bbox_area > 0 else 0
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        area_fraction = area / image_area
        # Una mancha de brillo real es pequeña-mediana, compacta (extent alto),
        # razonablemente redonda (no una franja delgada de texto) y no ocupa
        # la mayoría de la imagen.
        if 0.01 <= area_fraction <= 0.15 and extent > 0.5 and aspect > 0.4:
            flags.append("POSSIBLE_GLARE")
            break

    return flags


# ---------------------------------------------------------------------
# API de alto nivel usada por main.py
# ---------------------------------------------------------------------
def process_auto(image_bytes: bytes):
    image = load_image_with_exif_correction(image_bytes)
    corners, confidence = detect_document_corners(image)

    if corners is None or confidence < 80:
        return {
            "manual_required": True,
            "confidence": confidence,
            "message": "Document edges could not be detected confidently.",
        }

    processed = warp_and_enhance(image, corners)
    flags = quality_checks(processed)

    success, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {
        "manual_required": False,
        "confidence": confidence,
        "quality_flags": flags,
        "processed_image_bytes": buf.tobytes() if success else None,
        "detected_corners": corners.tolist(),
    }


def process_manual(image_bytes: bytes, corners_list: list):
    image = load_image_with_exif_correction(image_bytes)
    corners = order_points(np.array(corners_list, dtype="float32"))
    processed = warp_and_enhance(image, corners)
    flags = quality_checks(processed)

    success, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {
        "quality_flags": flags,
        "processed_image_bytes": buf.tobytes() if success else None,
    }
