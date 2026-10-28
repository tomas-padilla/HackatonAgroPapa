"""Pipeline de procesamiento de imagenes NIR para papas."""

from auto_rotation import auto_enderezar_surcos
from furrow_filter import filtrar_maleza_por_surco
from peak_extractor import extraer_centroides_maximos_locales
from pipeline import PipelineResult, detect_plants, procesar_fragmento_lote, process_image

__all__ = [
	"PipelineResult",
	"auto_enderezar_surcos",
	"detect_plants",
	"extraer_centroides_maximos_locales",
	"filtrar_maleza_por_surco",
	"procesar_fragmento_lote",
	"process_image",
]
