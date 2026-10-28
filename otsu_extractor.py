"""Extraccion de candidatos mediante umbral de Otsu y componentes conectadas."""

from __future__ import annotations

import cv2
import numpy as np


def extract_candidate_centroids(
    image: np.ndarray,
    min_area: int = 20,
    max_area: int | None = None,
    open_kernel_size: int = 3,
    invert: bool = False,
) -> list[tuple[float, float]]:
    """Devuelve centroides de objetos segmentados con Otsu.

    ``min_area`` y ``max_area`` eliminan ruido y regiones demasiado grandes.
    Las coordenadas se expresan como ``(x, y)`` en pixeles.
    """
    if min_area < 1:
        raise ValueError("min_area debe ser mayor que cero.")
    if max_area is not None and max_area < min_area:
        raise ValueError("max_area debe ser mayor o igual que min_area.")
    gray = _to_grayscale(image)
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, threshold_type | cv2.THRESH_OTSU)

    if open_kernel_size > 1:
        if open_kernel_size % 2 == 0:
            raise ValueError("open_kernel_size debe ser impar.")
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    candidates = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area and (max_area is None or area <= max_area):
            x, y = centroids[label]
            candidates.append((float(x), float(y)))
    return candidates


def extraer_centroides_otsu(
    img_rotada: np.ndarray,
    area_min: int = 25,
    area_max: int | None = 100000,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Devuelve la mascara Otsu y los centroides dentro del rango de area."""
    if area_min < 1:
        raise ValueError("area_min debe ser mayor que cero.")
    if area_max is not None and area_max < area_min:
        raise ValueError("area_max debe ser mayor o igual que area_min.")
    gray = _to_grayscale(img_rotada)
    _, mascara = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mascara, 8)
    candidatos = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= area_min and (area_max is None or area <= area_max):
            x, y = centroids[label]
            candidatos.append((float(x), float(y)))
    return mascara, candidatos


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("La imagen debe ser un ndarray no vacio.")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(image, code)
    raise ValueError("La imagen debe ser escala de grises o tener 3/4 canales.")
