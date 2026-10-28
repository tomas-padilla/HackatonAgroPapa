"""Punto de entrada de ejemplo para el pipeline de papas NIR."""

from __future__ import annotations

import argparse
import json
import os

import cv2

from pipeline import procesar_fragmento_lote


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecta plantas de papa en una imagen NIR.")
    parser.add_argument(
        "image",
        nargs="?",
        default="Imagenes/DJI_20251018171940_0351_MS_NIR_verificacion.png",
    )
    parser.add_argument("--min-distancia-plantas-px", type=int, default=24)
    parser.add_argument("--umbral-relativo-brillo", type=float, default=0.28)
    parser.add_argument("--umbral-absoluto-brillo", type=float, default=50.0)
    parser.add_argument("--margen-borde-px", type=int, default=40)
    parser.add_argument("--ancho-pasillo-px", type=int, default=22)
    parser.add_argument("--json", dest="json_path", default="coordenadas_plantas.json")
    parser.add_argument("--sin-json", action="store_true")
    args = parser.parse_args()

    coordenadas = procesar_fragmento_lote(
        args.image,
        min_distancia_plantas_px=args.min_distancia_plantas_px,
        umbral_relativo_brillo=args.umbral_relativo_brillo,
        umbral_absoluto_brillo=args.umbral_absoluto_brillo,
        margen_borde_px=args.margen_borde_px,
        ancho_pasillo_px=args.ancho_pasillo_px,
    )
    print(f"Plantas halladas: {len(coordenadas)}")
    print(f"Primeras 10 coordenadas: {coordenadas[:10]}")
    if not args.sin_json:
        with open(args.json_path, "w", encoding="utf-8") as archivo:
            json.dump(coordenadas, archivo, ensure_ascii=False, indent=2)
        print(f"Coordenadas guardadas en: {os.path.abspath(args.json_path)}")


if __name__ == "__main__":
    main()
