"""Filtrado geometrico de centroides para conservar plantas en surcos."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import cv2


def filtrar_maleza_por_surco(
    mascara_binaria: np.ndarray,
    candidatos_coords: Sequence[tuple[float, float]],
    ancho_pasillo_px: int = 25,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Clasifica candidatos usando una mascara morfologica de pasillos verticales."""
    if not isinstance(mascara_binaria, np.ndarray) or mascara_binaria.ndim != 2:
        raise ValueError("La imagen o mascara debe ser una matriz bidimensional.")
    if ancho_pasillo_px < 1:
        raise ValueError("ancho_pasillo_px debe ser mayor que cero.")

    imagen = mascara_binaria.astype(np.uint8, copy=False)
    if np.unique(imagen).size > 2:
        # La imagen peak-preprocessed no es binaria: segmentamos solo las
        # zonas brillantes para construir una guia de pasillos verticales.
        threshold = max(
            float(cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]),
            float(np.percentile(imagen, 75)),
        )
        _, mascara = cv2.threshold(imagen, threshold, 255, cv2.THRESH_BINARY)
    else:
        mascara = np.where(imagen > 0, 255, 0).astype(np.uint8)
    cierre_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    conectada = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, cierre_kernel)
    ancho = 2 * ancho_pasillo_px + 1
    dilatacion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ancho, 1))
    pasillos = cv2.dilate(conectada, dilatacion_kernel)

    plantas_validas = []
    malezas_descartadas = []
    height, width = pasillos.shape
    for x, y in candidatos_coords:
        x_int, y_int = int(round(x)), int(round(y))
        punto = (0 <= x_int < width and 0 <= y_int < height and pasillos[y_int, x_int] > 0)
        coordenada = (float(x), float(y))
        (plantas_validas if punto else malezas_descartadas).append(coordenada)
    return plantas_validas, malezas_descartadas


def filter_by_furrows(
    centroids: Sequence[tuple[float, float]],
    image_shape: tuple[int, int],
    furrow_angle: float = 0.0,
    row_spacing: float | None = None,
    row_tolerance: float | None = None,
    min_plants_per_row: int = 1,
) -> list[tuple[float, float]]:
    """Conserva centroides proximos a lineas paralelas de surcos.

    La geometria se evalua en coordenadas rotadas: el eje ``u`` sigue el
    surco y el eje ``v`` es perpendicular. Si no se informa ``row_spacing``,
    se estima a partir de la distribucion de candidatos.
    """
    height, width = image_shape[:2]
    if height < 1 or width < 1:
        raise ValueError("image_shape debe tener dimensiones positivas.")
    if row_spacing is not None and row_spacing <= 0:
        raise ValueError("row_spacing debe ser mayor que cero.")
    if row_tolerance is not None and row_tolerance <= 0:
        raise ValueError("row_tolerance debe ser mayor que cero.")
    if min_plants_per_row < 1:
        raise ValueError("min_plants_per_row debe ser mayor que cero.")
    if not centroids:
        return []

    points = np.asarray(centroids, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("centroids debe contener pares (x, y).")

    radians = np.deg2rad(furrow_angle)
    normal = np.array([-np.sin(radians), np.cos(radians)])
    center = np.array([width / 2.0, height / 2.0])
    normal_coordinates = (points - center) @ normal
    if furrow_angle == 0.0:
        # Tras el enderezado, los surcos son verticales y se separan en X.
        normal_coordinates = points[:, 0] - center[0]

    spacing = row_spacing or _estimate_spacing(normal_coordinates)
    tolerance = row_tolerance or max(3.0, spacing * 0.30)
    row_origin = float(np.min(normal_coordinates))
    nearest_row = (
        np.round((normal_coordinates - row_origin) / spacing) * spacing
        + row_origin
    )
    residual = np.abs(normal_coordinates - nearest_row)
    accepted = residual <= tolerance

    row_ids = np.round(
        (normal_coordinates[accepted] - row_origin) / spacing
    ).astype(int)
    valid_rows = {
        row_id
        for row_id in np.unique(row_ids)
        if int(np.sum(row_ids == row_id)) >= min_plants_per_row
    }
    result = [
        (float(point[0]), float(point[1]))
        for point, keep, row_id in zip(
            points,
            accepted,
            np.round((normal_coordinates - row_origin) / spacing).astype(int),
        )
        if keep and row_id in valid_rows
    ]
    return result


def _estimate_spacing(values: np.ndarray) -> float:
    if len(values) < 2:
        return max(10.0, float(np.ptp(values)) or 10.0)
    sorted_values = np.sort(values)
    gaps = np.diff(sorted_values)
    positive_gaps = gaps[gaps > 2.0]
    if len(positive_gaps) == 0:
        return max(10.0, float(np.ptp(values)) / max(1, len(values) - 1))
    return max(10.0, float(np.median(positive_gaps)))
