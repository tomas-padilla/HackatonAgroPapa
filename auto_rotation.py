"""Correccion automatica de inclinacion para imagenes NIR."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def estimate_rotation_angle(
    image: np.ndarray,
    canny_thresholds: Tuple[int, int] = (50, 150),
    hough_threshold: int = 80,
    min_line_length: int | None = None,
    max_line_gap: int = 20,
) -> float:
    """Estima en grados la inclinacion dominante de las lineas de los surcos.

    El resultado se normaliza al rango [-45, 45], donde cero representa
    surcos horizontales. Si no se detectan lineas, devuelve 0.0.
    """
    gray = _to_grayscale(image)
    # La mascara preliminar reduce el efecto del fondo antes de buscar surcos.
    _, vegetation = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(vegetation, canny_thresholds[0], canny_thresholds[1])
    height, width = gray.shape
    minimum_length = min_line_length or max(20, int(min(height, width) * 0.25))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=minimum_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        return 0.0

    segments = np.asarray(lines).reshape(-1, 4)
    angles = []
    for x1, y1, x2, y2 in segments:
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        normalized = angle % 180.0
        if normalized > 90.0:
            normalized -= 180.0
        angles.append(normalized)

    if not angles:
        return 0.0
    return float(np.median(angles))


def rotate_image(image: np.ndarray, angle: float, border_value: int = 0) -> np.ndarray:
    """Rota una imagen conservando todo el contenido y su tamano."""
    _validate_image(image)
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def auto_rotate(image: np.ndarray, **kwargs: object) -> tuple[np.ndarray, float]:
    """Estima el angulo y devuelve ``(imagen_rotada, angulo_aplicado)``."""
    inclination = estimate_rotation_angle(image, **kwargs)
    if inclination == 0.0:
        return image.copy(), 0.0
    applied_angle = 90.0 - inclination
    return rotate_image(image, applied_angle), applied_angle


def auto_enderezar_surcos(img_nir: np.ndarray) -> tuple[np.ndarray, float]:
    """Endereza los surcos verticalmente y devuelve la imagen y el giro aplicado."""
    return auto_rotate(img_nir)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    _validate_image(image)
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("La imagen debe ser un ndarray no vacio.")
    if image.ndim not in (2, 3):
        raise ValueError("La imagen debe tener 2 o 3 dimensiones.")
