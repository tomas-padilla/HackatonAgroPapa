"""Extraccion de plantas mediante maximos locales de intensidad NIR."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from skimage.feature import peak_local_max


def extraer_centroides_maximos_locales(
    img_rotada: np.ndarray,
    min_distancia_px: int = 24,
    umbral_relativo: float = 0.28,
    umbral_absoluto: float = 50.0,
    margen_borde_px: int = 40,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Preprocesa la imagen y devuelve coordenadas de picos locales.

    Las coordenadas conservan el sistema espacial de la imagen original.
    """
    if min_distancia_px < 1:
        raise ValueError("min_distancia_px debe ser mayor que cero.")
    if not 0.0 <= umbral_relativo <= 1.0:
        raise ValueError("umbral_relativo debe estar entre 0 y 1.")
    if umbral_absoluto < 0:
        raise ValueError("umbral_absoluto no puede ser negativo.")
    if margen_borde_px < 0:
        raise ValueError("margen_borde_px no puede ser negativo.")

    gray = _to_grayscale(img_rotada)
    height, width = gray.shape
    if margen_borde_px * 2 >= height or margen_borde_px * 2 >= width:
        raise ValueError("margen_borde_px deja un ROI vacio.")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    processed = cv2.GaussianBlur(enhanced, (7, 7), 0)
    roi = processed[
        margen_borde_px : height - margen_borde_px,
        margen_borde_px : width - margen_borde_px,
    ]

    coordinates = peak_local_max(
        roi,
        min_distance=min_distancia_px,
        threshold_rel=umbral_relativo,
        threshold_abs=umbral_absoluto,
        exclude_border=False,
    )
    candidates = [
        (float(x + margen_borde_px), float(y + margen_borde_px))
        for y, x in coordinates
    ]
    return processed, candidates


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("La imagen debe ser un ndarray no vacio.")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(image, code)
    else:
        raise ValueError("La imagen debe ser escala de grises o tener 3/4 canales.")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray
