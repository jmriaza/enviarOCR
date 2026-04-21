# OCR + Layout Analysis v1.0

Pipeline de procesamiento OCR por lotes, robusto y resiliente, con soporte
para analisis de layout, preprocesamiento avanzado de imagen (CLAHE, denoise,
threshold) y gestion automatica de documentos grandes.

## Caracteristicas principales

- Procesamiento OCR en batch sobre multiples ficheros PDF
- Analisis de layout por pagina
- Preprocesamiento de imagen configurable:
  - CLAHE en espacio de color LAB
  - Reduccion de ruido (denoise)
  - Umbral de binarizacion (threshold) con modo configurable via CLI
- Gestion automatica de documentos grandes (`DecompressionBombError`)
- Movimiento automatico de ficheros a carpetas `completados` y `bigsizeDocuments`
- Resumen final tabulado con estadisticas de exito, parciales, fallos y bigsize
- Workspace temporal para procesamiento seguro
- Output quality configurable
- OCR garantizado a un DPI minimo

---

## Requisitos

- Python 3.9 o superior
- Tesseract OCR instalado en el sistema
- Dependencias Python (ver `requirements.txt`):


pip install -r requirements.txt
### Dependencias principales

| Paquete | Uso |
|---|---|
| `Pillow` | Manipulacion de imagenes y control de `MAX_IMAGE_PIXELS` |
| `pytesseract` | Motor OCR |
| `opencv-python` | CLAHE, denoise, threshold |
| `pdf2image` | Conversion PDF a imagenes |
| `numpy` | Procesamiento matricial de imagenes |

---

## Instalacion

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/ocr-layout-analysis.git
cd ocr-layout-analysis

# 2. Crear entorno virtual
python3 -m virualenv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar que Tesseract esta instalado
tesseract --version


## Uso

# Sintaxis basica
python3 enviarOCR.py <fichero_o_directorio> -o <directorio_salida> [opciones]

Flag	Descripcion	Valor por defecto
-o, --output	    Directorio base de salida	Obligatorio
--use-lab	        Activa CLAHE en espacio de color LAB	Desactivado
--no-denoise	    Desactiva la reduccion de ruido	Activado
--threshold-mode	Modo de umbral: auto, otsu, none	auto
--output-quality	Calidad del output (1-100)	85


## Ejemplos de uso

# Procesamiento estandar de un PDF
python3 enviarOCR.py documento.pdf -o procesados/

# Documentos sepias o historicos (fondo preservado, sin threshold)
python3 enviarOCR.py doc.pdf -o procesados/ \
  --use-lab --threshold-mode none

# Batch completo de un directorio con alta calidad
python3 enviarOCR.py ./pdfs/ -o ./resultados/ \
  --use-lab --output-quality 95

# Sin denoise, threshold Otsu
python3 enviarOCR.py doc.pdf -o salida/ \
  --no-denoise --threshold-mode otsu
  

## Estructura de salida

salida/
├── completados/          # PDFs procesados correctamente
│   ├── documento_01.pdf
│   └── documento_02.pdf
├── bigsizeDocuments/     # PDFs rechazados por exceder MAX_IMAGE_PIXELS
│   └── documento_grande.pdf
└── logs/                 # Logs detallados por sesion
    └── ocr_20260421.log  
	

## Estados de procesamiento

Estado		Descripcion
SUCCESS	OCR completado sin errores
PARTIAL	OCR completado con advertencias en algunos pasos
FAILED		El procesamiento fallo en uno o mas pasos criticos
BIGSIZE		Fichero demasiado grande (DecompressionBombError)


## Resumen de consola

Al finalizar el batch, se muestra una tabla detallada con:
Numero de fichero y nombre
Estado final (SUCCESS, PARTIAL, FAILED, BIGSIZE)
Directorio de destino
Tiempo de procesamiento individual
Numero de paginas procesadas
Palabras detectadas por el OCR
Detalle del error (si aplica)

## Seguido de los totales globales:

TOTALES:  42 exitosos (84%)  |  5 parciales (10%)  |  2 fallidos (4%)  |  1 bigsize (2%)  |  Total: 50
50 fichero(s) -> completados  (./completados)
1  fichero(s) -> bigsizeDocuments  (./bigsizeDocuments)
Tiempo total: 312.47s (5.2 min)


## Configuracion avanzada

# MAX_IMAGE_PIXELS
Controla el limite de pixels permitido antes de lanzar DecompressionBombError.
Se puede ajustar directamente en el script:Image.
MAX_IMAGE_PIXELS = 300_000_000  # 300 megapixels

# OCR_MIN_DPI
Define el DPI minimo garantizado para el motor OCR. Ficheros con resolucion
inferior seran reescalados automaticamente antes del procesamiento:
OCR_MIN_DPI = 300


## Logs
El sistema genera logs detallados por sesion con:
Cabecera con la configuracion activa del batch
Progreso fichero a fichero
Detalle de errores por paso
Separadores visuales para facilitar la lectura
