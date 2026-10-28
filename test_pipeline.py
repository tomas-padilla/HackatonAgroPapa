"""Pruebas basicas ejecutables sin imagen real."""

import cv2
import numpy as np

from pipeline import detect_plants


def test_empty_image_returns_no_centroids() -> None:
    image = np.zeros((120, 160), dtype=np.uint8)
    assert detect_plants(image) == []


def test_detects_components_on_a_row() -> None:
    image = np.zeros((120, 160), dtype=np.uint8)
    for x in (50, 80, 110):
        cv2.circle(image, (x, 60), 6, 255, -1)

    centroids = detect_plants(
        image,
        row_spacing=30,
        row_tolerance=8,
        min_distancia_plantas_px=15,
        min_area=40,
    )

    assert len(centroids) == 3
    assert all(abs(y - 60) <= 4 for _, y in centroids)
