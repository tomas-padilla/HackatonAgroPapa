"""Verificacion visual temporal del pipeline NIR."""

from pathlib import Path

import cv2

from auto_rotation import auto_enderezar_surcos
from furrow_filter import filtrar_maleza_por_surco
from peak_extractor import extraer_centroides_maximos_locales


RUTA_ENTRADA = Path("Imagenes/DJI_20251018171940_0351_MS_NIR.TIF")
RUTA_SALIDA = RUTA_ENTRADA.with_name(f"{RUTA_ENTRADA.stem}_verificacion.png")


def main() -> None:
    imagen = cv2.imread(str(RUTA_ENTRADA), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo leer: {RUTA_ENTRADA}")

    imagen_rotada, angulo = auto_enderezar_surcos(imagen)
    procesada, candidatos = extraer_centroides_maximos_locales(
        imagen_rotada,
        min_distancia_px=24,
        umbral_relativo=0.28,
        umbral_absoluto=50.0,
        margen_borde_px=40,
    )
    plantas_validas, malezas = filtrar_maleza_por_surco(
        procesada,
        candidatos,
        ancho_pasillo_px=22,
    )

    resultado = cv2.cvtColor(imagen_rotada, cv2.COLOR_GRAY2BGR)
    for numero, (x, y) in enumerate(plantas_validas, start=1):
        punto = (round(x), round(y))
        cv2.circle(resultado, punto, 8, (0, 0, 255), 2)
        cv2.putText(
            resultado,
            str(numero),
            (punto[0] + 8, punto[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(str(RUTA_SALIDA), resultado):
        raise OSError(f"No se pudo guardar: {RUTA_SALIDA}")

    print(f"Angulo aplicado: {angulo:.2f} grados")
    print(f"Candidatos detectados: {len(candidatos)}")
    print(f"Plantas validas: {len(plantas_validas)}")
    print(f"Malezas descartadas: {len(malezas)}")
    print(f"Imagen guardada en: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
