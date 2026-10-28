"""Coordinador del pipeline NIR para plantas de papa."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

from auto_rotation import auto_enderezar_surcos, auto_rotate
from furrow_filter import filtrar_maleza_por_surco, filter_by_furrows
from peak_extractor import extraer_centroides_maximos_locales


@dataclass(frozen=True)
class PipelineResult:
    """Resultado intermedio y final del procesamiento."""

    centroids: list[tuple[float, float]]
    candidates: list[tuple[float, float]]
    rotation_angle: float
    rotated_image: np.ndarray


def process_image(
    image: np.ndarray,
    *,
    row_spacing: float | None = None,
    row_tolerance: float | None = None,
    min_distancia_plantas_px: int = 24,
    umbral_relativo_brillo: float = 0.28,
    umbral_absoluto_brillo: float = 50.0,
    margen_borde_px: int = 40,
    ancho_pasillo_px: int = 22,
    min_area: int | None = None,
    max_area: int | None = None,
    min_plants_per_row: int = 1,
    rotation_kwargs: dict[str, object] | None = None,
) -> PipelineResult:
    """Ejecuta rotacion, segmentacion y filtrado de surcos."""
    rotated, angle = auto_rotate(image, **(rotation_kwargs or {}))
    processed, candidates = extraer_centroides_maximos_locales(
        rotated,
        min_distancia_px=min_distancia_plantas_px,
        umbral_relativo=umbral_relativo_brillo,
        umbral_absoluto=umbral_absoluto_brillo,
        margen_borde_px=margen_borde_px,
    )
    centroids, _ = filtrar_maleza_por_surco(
        processed,
        candidates,
        ancho_pasillo_px=ancho_pasillo_px,
    )
    return PipelineResult(centroids, candidates, angle, rotated)


def detect_plants(image: np.ndarray, **kwargs: object) -> list[tuple[float, float]]:
    """Atajo que devuelve solo la lista limpia de coordenadas."""
    return process_image(image, **kwargs).centroids


def procesar_fragmento_lote(
    ruta_imagen: str,
    min_distancia_plantas_px: int = 24,
    umbral_relativo_brillo: float = 0.28,
    umbral_absoluto_brillo: float = 50.0,
    margen_borde_px: int = 40,
    ancho_pasillo_px: int = 22,
) -> list[tuple[float, float]]:
    """Procesa una imagen NIR desde disco y devuelve las plantas validas."""
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta_imagen}")
    img_rotada, _ = auto_enderezar_surcos(imagen)
    procesada, candidatos = extraer_centroides_maximos_locales(
        img_rotada,
        min_distancia_px=min_distancia_plantas_px,
        umbral_relativo=umbral_relativo_brillo,
        umbral_absoluto=umbral_absoluto_brillo,
        margen_borde_px=margen_borde_px,
    )
    plantas_validas, _ = filtrar_maleza_por_surco(
        procesada, candidatos, ancho_pasillo_px
    )
    return plantas_validas
