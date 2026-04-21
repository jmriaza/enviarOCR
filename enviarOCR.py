#!/usr/bin/env python3

import os
import gc
import sys
import json
import glob
import ctypes
import shutil
import logging
import re
import traceback
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import cv2

# ============================================================================
# CONFIGURACION DE PILLOW — CALIBRADO PARA 16 GB RAM
# ============================================================================
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1_200_000_000

from pdf2image import convert_from_path

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    from pypdf import PdfWriter, PdfReader
    HAS_PYPDF = True
except ImportError:
    try:
        from PyPDF2 import PdfWriter, PdfReader
        HAS_PYPDF = True
    except ImportError:
        HAS_PYPDF = False


# ============================================================================
# CONSTANTES DE REDIMENSIONAMIENTO
# ============================================================================

OCR_MIN_DPI            = 300
SAFE_PIXEL_LIMIT       = 800_000_000
ABSOLUTE_PIXEL_LIMIT   = 1_200_000_000

# — modos de threshold válidos
THRESHOLD_MODES = ("none", "binary")


# ============================================================================
# STATUS DE PROCESAMIENTO
# ============================================================================

class ProcessStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED  = "FAILED"
    SKIPPED = "SKIPPED"
    PARTIAL = "PARTIAL"
    BIGSIZE = "BIGSIZE"


@dataclass
class FileResult:
    pdf_name:             str
    pdf_path:             str
    status:               ProcessStatus
    output_dir:           str            = ""
    steps_ok:             int            = 0
    steps_total:          int            = 4
    processing_time:      float          = 0.0
    error_summary:        str            = ""
    failed_steps:         List[str]      = None
    files_generated:      List[str]      = None
    statistics:           Dict[str, Any] = None
    moved_to_completed:   bool           = False
    completed_path:       str            = ""
    moved_to_bigsize:     bool           = False
    bigsize_path:         str            = ""

    def __post_init__(self):
        if self.failed_steps    is None: self.failed_steps    = []
        if self.files_generated is None: self.files_generated = []
        if self.statistics      is None: self.statistics      = {}

    def to_dict(self) -> Dict:
        return {
            "pdf_name":            self.pdf_name,
            "pdf_path":            self.pdf_path,
            "status":              self.status.value,
            "output_dir":          self.output_dir,
            "steps_ok":            self.steps_ok,
            "steps_total":         self.steps_total,
            "processing_time":     round(self.processing_time, 2),
            "error_summary":       self.error_summary,
            "failed_steps":        self.failed_steps,
            "files_generated":     self.files_generated,
            "statistics":          self.statistics,
            "moved_to_completed":  self.moved_to_completed,
            "completed_path":      self.completed_path,
            "moved_to_bigsize":    self.moved_to_bigsize,
            "bigsize_path":        self.bigsize_path
        }


# ============================================================================
# GESTOR DE DIRECTORIO TEMPORAL
# ============================================================================

class TemporaryWorkspace:
    def __init__(self, pdf_name: str, logger: Optional["ProcessingLogger"] = None):
        self.pdf_name = pdf_name
        self.logger   = logger
        temp_base = Path(tempfile.gettempdir()) / "easyocr_workspace"
        temp_base.mkdir(parents=True, exist_ok=True)
        timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.workspace = temp_base / f"pdf_{timestamp}"
        self.workspace.mkdir(parents=True, exist_ok=True)
        if logger:
            logger.log_debug(f"Workspace temporal creado: {self.workspace}")

    def get_page_path(self, page_num: int, extension: str = "png") -> Path:
        return self.workspace / f"page_{page_num:04d}.{extension}"

    def get_all_pages(self, extension: str = "png") -> List[Path]:
        return sorted(self.workspace.glob(f"page_*.{extension}"))

    def cleanup(self):
        if self.workspace.exists():
            try:
                shutil.rmtree(self.workspace)
                if self.logger:
                    self.logger.log_debug(
                        f"Workspace temporal limpiado: {self.workspace}"
                    )
            except Exception as exc:
                if self.logger:
                    self.logger.log_warning(f"Error limpiando workspace: {exc}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# ============================================================================
# LIBERACIÓN DE MEMORIA
# ============================================================================

def release_memory(
    logger: Optional["ProcessingLogger"] = None
) -> Dict[str, float]:
    freed     = {}
    collected = gc.collect(generation=2)
    freed["gc_objects_collected"] = collected
    try:
        if sys.platform.startswith("linux"):
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
            freed["malloc_trim"] = True
        elif sys.platform == "darwin":
            libc = ctypes.CDLL("libSystem.B.dylib")
            freed["malloc_trim"] = True
    except Exception:
        freed["malloc_trim"] = False
    gc.collect(generation=2)
    if logger:
        logger.log_debug(
            f"Memoria liberada — GC objetos recogidos: {collected} | "
            f"malloc_trim: {freed.get('malloc_trim', False)}"
        )
    return freed


# ============================================================================
# GESTIÓN DE CARPETAS ESPECIALES
# ============================================================================

def ensure_special_folder(base_dir: str, folder_name: str) -> Path:
    folder_path = Path(base_dir) / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def move_file_to_folder(
    pdf_path:   str,
    target_dir: Path,
    reason:     str = "",
    logger:     Optional["ProcessingLogger"] = None
) -> Tuple[bool, str]:
    src = Path(pdf_path)
    if not src.exists():
        if logger:
            logger.log_warning(
                f"No se puede mover '{src.name}': fichero no encontrado."
            )
        return False, ""
    dst = target_dir / src.name
    if dst.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = target_dir / f"{src.stem}_{timestamp}{src.suffix}"
        if logger:
            logger.log_warning(
                f"Colision de nombre en '{target_dir.name}' — "
                f"renombrado a '{dst.name}'"
            )
    try:
        shutil.move(str(src), str(dst))
        if logger:
            msg = f"Fichero movido a '{target_dir.name}'"
            if reason:
                msg += f" ({reason})"
            msg += f": {dst.name}"
            logger.log_success(msg)
        return True, str(dst)
    except (PermissionError, shutil.Error, OSError) as exc:
        if logger:
            logger.log_error(
                f"Error moviendo '{src.name}' a '{target_dir.name}': {exc}"
            )
        return False, ""


# ============================================================================
# UTILIDADES DE NOMENCLATURA Y RUTAS
# ============================================================================

def sanitize_filename(name: str) -> str:
    name = Path(name).stem
    name = re.sub(r'[^\w\-]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_').lower()


def build_output_dir(base_dir: str, pdf_filename: str) -> Path:
    clean_name  = sanitize_filename(pdf_filename)
    output_path = Path(base_dir) / f"output_{clean_name}"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def build_output_filename(prefix: str, pdf_filename: str, extension: str) -> str:
    clean_name = sanitize_filename(pdf_filename)
    return f"{prefix}_{clean_name}.{extension.lstrip('.')}"


def expand_pdf_sources(sources: List[str]) -> List[str]:
    all_pdfs = []
    for source in sources:
        source_path = Path(source)
        if source_path.is_dir():
            all_pdfs.extend(
                [str(p) for p in sorted(source_path.glob("*.pdf"))]
            )
        elif "*" in source or "?" in source:
            all_pdfs.extend(sorted(glob.glob(source)))
        elif source_path.exists():
            all_pdfs.append(str(source_path))
        else:
            expanded = glob.glob(source)
            if expanded:
                all_pdfs.extend(sorted(expanded))
    seen   = set()
    unique = []
    for p in all_pdfs:
        if not Path(p).exists() or Path(p).suffix.lower() != ".pdf":
            continue
        resolved = str(Path(p).resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)
    return unique


# ============================================================================
# LOGGING
# ============================================================================

class ProcessingLogger:
    COLORS = {
        "DEBUG":   "\033[36m",
        "INFO":    "\033[34m",
        "SUCCESS": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR":   "\033[31m",
        "HEADER":  "\033[35m",
        "BIGSIZE": "\033[95m",
        "BOLD":    "\033[1m",
    }
    RESET = "\033[0m"

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"processing_{timestamp}.log"
        self._setup_logger()

    def _setup_logger(self):
        self._logger = logging.getLogger(f"EasyOCR_{id(self)}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers = []
        fh = logging.FileHandler(str(self.log_file), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self._logger.addHandler(fh)

    def _print(self, level: str, message: str):
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = self.COLORS.get(level, "")
        print(f"{color}[{ts}] [{level:7s}] {message}{self.RESET}")
        lvl = logging.INFO if level in ("SUCCESS", "HEADER", "BIGSIZE") else \
              getattr(logging, level, logging.INFO)
        self._logger.log(lvl, message)

    def log_debug(self, msg):   self._print("DEBUG",   msg)
    def log_info(self, msg):    self._print("INFO",    msg)
    def log_success(self, msg): self._print("SUCCESS", msg)
    def log_warning(self, msg): self._print("WARNING", msg)
    def log_error(self, msg):   self._print("ERROR",   msg)
    def log_header(self, msg):  self._print("HEADER",  msg)
    def log_bigsize(self, msg): self._print("BIGSIZE", msg)

    def separator(self, char: str = "=", length: int = 78):
        line = char * length
        print(f"{self.COLORS['BOLD']}{line}{self.RESET}")
        self._logger.info(line)

    def blank(self):
        print()
        self._logger.info("")


# ============================================================================
# REDIMENSIONAMIENTO VARIABLE
# ============================================================================

def compute_safe_resize(
    original_width:   int,
    original_height:  int,
    original_dpi:     int,
    min_ocr_dpi:      int = OCR_MIN_DPI,
    safe_pixel_limit: int = SAFE_PIXEL_LIMIT,
    logger: Optional[ProcessingLogger] = None
) -> Tuple[int, int, int, bool]:
    total_pixels = original_width * original_height
    if total_pixels <= safe_pixel_limit:
        return original_width, original_height, original_dpi, False
    scale_by_pixels = (safe_pixel_limit / total_pixels) ** 0.5
    scale_min_dpi   = (
        min_ocr_dpi / original_dpi if original_dpi > min_ocr_dpi else 1.0
    )
    final_scale   = max(scale_by_pixels, scale_min_dpi)
    effective_dpi = int(original_dpi * final_scale)
    new_width  = max(1, int(original_width  * final_scale))
    new_height = max(1, int(original_height * final_scale))
    new_pixels = new_width * new_height
    if logger:
        logger.log_warning(
            f"Redimensionamiento variable activado: "
            f"{original_width}x{original_height} ({total_pixels:,} px) → "
            f"{new_width}x{new_height} ({new_pixels:,} px) | "
            f"Escala: {final_scale:.3f} | DPI efectivo: {effective_dpi}"
        )
        if effective_dpi < min_ocr_dpi:
            logger.log_warning(
                f"AVISO: DPI efectivo ({effective_dpi}) < minimo ({min_ocr_dpi})."
            )
    return new_width, new_height, effective_dpi, True


def safe_resize_image(
    img_array:    np.ndarray,
    original_dpi: int,
    logger:       Optional[ProcessingLogger] = None
) -> Tuple[np.ndarray, int, bool]:
    height, width = img_array.shape[:2]
    new_w, new_h, eff_dpi, was_resized = compute_safe_resize(
        original_width=width, original_height=height,
        original_dpi=original_dpi, min_ocr_dpi=OCR_MIN_DPI,
        safe_pixel_limit=SAFE_PIXEL_LIMIT, logger=logger
    )
    if not was_resized:
        return img_array, original_dpi, False
    resized = cv2.resize(img_array, (new_w, new_h),
                         interpolation=cv2.INTER_LANCZOS4)
    return resized, eff_dpi, True


# ============================================================================
# CLAHE EN ESPACIO LAB 
# ============================================================================

def apply_clahe_lab(
    img_bgr:   np.ndarray,
    clip:      float = 2.0,
    tile_size: Tuple[int, int] = (8, 8),
    logger:    Optional[ProcessingLogger] = None
) -> np.ndarray:
    """
    CLAHE solo sobre canal L (luminosidad) en espacio LAB.
    Preserva los tonos de color originales (sepia, amarillo, marrón).
    """
    lab                             = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe_obj                       = cv2.createCLAHE(
        clipLimit=clip, tileGridSize=tile_size
    )
    l_equalized   = clahe_obj.apply(l_channel)
    lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
    result_bgr    = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
    if logger:
        logger.log_debug(
            f"CLAHE-LAB: clip={clip}, tile={tile_size} — "
            f"canal L ecualizado, color preservado"
        )
    return result_bgr


# ============================================================================
# MODELOS DE DATOS
# ============================================================================

@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class TextBlock:
    text:       str
    confidence: float
    bbox:       BoundingBox
    block_type: str = "text"
    page_num:   int = 1
    block_id:   str = ""

    def to_dict(self) -> Dict:
        return {
            "text":       self.text,
            "confidence": self.confidence,
            "bbox":       self.bbox.to_dict(),
            "block_type": self.block_type,
            "page_num":   self.page_num,
            "block_id":   self.block_id
        }


@dataclass
class PageStructure:
    page_num:        int
    width:           int
    height:          int
    blocks:          List[TextBlock]
    full_text:       str
    processing_time: float

    def to_dict(self) -> Dict:
        return {
            "page_num":        self.page_num,
            "width":           self.width,
            "height":          self.height,
            "total_blocks":    len(self.blocks),
            "full_text":       self.full_text,
            "blocks":          [b.to_dict() for b in self.blocks],
            "processing_time": self.processing_time
        }


# ============================================================================
# MEJORA DE CALIDAD
# ============================================================================

def enhance_pdf_quality_v5_7(
    pdf_path:                 str,
    output_path:              str,
    dpi:                      int   = 400,
    denoise_strength:         int   = 10,
    clahe_clip:               float = 3.0,
    adaptive_threshold_block: int   = 21,
    adaptive_threshold_c:     int   = 10,
    threshold_mode:           str   = "none",   
    use_lab:                  bool  = False,
    output_quality:           int   = 72,
    logger: Optional[ProcessingLogger] = None
) -> str:
    """
    Mejora la calidad del PDF procesando páginas a disco temporal.

    Parámetro threshold_mode ):
        'none'   : SIN binarización — conserva tonos sepia/grises/color.
                   Recomendado para documentos históricos, sepia, fondos
                   coloreados. La imagen se guarda tal cual sale del CLAHE
                   + denoise, sin destruir los tonos intermedios.
        'binary' : Aplica adaptiveThreshold + morfología (comportamiento
                   v5.7.3). Para documentos modernos con fondo blanco puro.

    Parámetro use_lab:
        False : GRAY → denoise → CLAHE → [threshold opcional]
        True  : BGR → CLAHE canal L (LAB) → GRAY → denoise → [threshold]

    Parámetro denoise_strength:
        0  : omite denoise completamente
        >0 : fastNlMeansDenoising con h=denoise_strength

    Puede lanzar Image.DecompressionBombError si supera MAX_IMAGE_PIXELS.
    """
    original_path = Path(pdf_path)
    output_path   = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not original_path.exists():
        raise FileNotFoundError(f"PDF no encontrado: {original_path}")

    if threshold_mode not in THRESHOLD_MODES:
        raise ValueError(
            f"threshold_mode inválido: '{threshold_mode}'. "
            f"Opciones: {THRESHOLD_MODES}"
        )

    if logger:
        mode_label     = "LAB (canal L)" if use_lab else "GRAY (clásico)"
        denoise_label  = f"h={denoise_strength}" if denoise_strength > 0 else "OMITIDO"
        thresh_label   = (
            "DESACTIVADO (tonos preservados)"
            if threshold_mode == "none"
            else f"adaptativo block={adaptive_threshold_block} C={adaptive_threshold_c}"
        )
        logger.log_info(
            f"Mejorando calidad del PDF ( CLAHE: {mode_label} | "
            f"denoise: {denoise_label} | threshold: {thresh_label} | "
            f"quality: {output_quality})..."
        )

    images = convert_from_path(str(original_path), dpi=dpi)

    if logger:
        logger.log_success(f"PDF convertido a {len(images)} imagen(s)")

    with TemporaryWorkspace(original_path.name, logger) as workspace:
        enhanced_paths = []

        for page_idx, image in enumerate(images, 1):
            if logger:
                logger.log_debug(
                    f"Procesando pagina {page_idx}/{len(images)} "
                    f"[CLAHE: {'LAB' if use_lab else 'GRAY'} | "
                    f"threshold: {threshold_mode}]..."
                )

            try:
                img_array = np.array(image)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_array

                # ============================================================
                # RAMA LAB  (--use-lab activado)
                # ============================================================
                if use_lab and len(img_bgr.shape) == 3:

                    img_bgr, effective_dpi, was_resized = safe_resize_image(
                        img_bgr, original_dpi=dpi, logger=logger
                    )
                    if was_resized and logger:
                        logger.log_info(
                            f"  Pag {page_idx}: redimensionada "
                            f"(DPI efectivo={effective_dpi})"
                        )

                    # CLAHE sobre canal L — preserva color/sepia
                    img_bgr_eq = apply_clahe_lab(
                        img_bgr, clip=clahe_clip,
                        tile_size=(8, 8), logger=logger
                    )

                    # BGR → GRAY para el resto del pipeline
                    gray = cv2.cvtColor(img_bgr_eq, cv2.COLOR_BGR2GRAY)

                    if denoise_strength > 0:
                        processed = cv2.fastNlMeansDenoising(
                            gray, h=denoise_strength,
                            templateWindowSize=7, searchWindowSize=21
                        )
                    else:
                        processed = gray
                        if logger:
                            logger.log_debug(
                                f"  Pag {page_idx}: denoise OMITIDO (h=0)"
                            )

                # ============================================================
                # RAMA GRAY CLÁSICA
                # ============================================================
                else:
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) \
                           if len(img_bgr.shape) == 3 else img_bgr

                    gray, effective_dpi, was_resized = safe_resize_image(
                        gray, original_dpi=dpi, logger=logger
                    )
                    if was_resized and logger:
                        logger.log_info(
                            f"  Pag {page_idx}: redimensionada "
                            f"(DPI efectivo={effective_dpi})"
                        )

                    if denoise_strength > 0:
                        denoised = cv2.fastNlMeansDenoising(
                            gray, h=denoise_strength,
                            templateWindowSize=7, searchWindowSize=21
                        )
                    else:
                        denoised = gray
                        if logger:
                            logger.log_debug(
                                f"  Pag {page_idx}: denoise OMITIDO (h=0)"
                            )

                    if clahe_clip > 0:
                        clahe_obj = cv2.createCLAHE(
                            clipLimit=clahe_clip, tileGridSize=(8, 8)
                        )
                        processed = clahe_obj.apply(denoised)
                    else:
                        processed = denoised

                # ============================================================
                # DESKEW — detecta ángulo en GRAY, corrige sobre 'processed'
                # ============================================================
                try:
                    coords = np.column_stack(np.where(processed > 100))
                    if len(coords) > 100:
                        angle = cv2.minAreaRect(coords)[-1]
                        if angle < -45:
                            angle = 90 + angle
                        if abs(angle) > 0.5:
                            h_img, w_img = processed.shape
                            center = (w_img // 2, h_img // 2)
                            M = cv2.getRotationMatrix2D(center, angle, 1.0)
                            processed = cv2.warpAffine(
                                processed, M, (w_img, h_img),
                                borderMode=cv2.BORDER_REPLICATE
                            )
                            if logger:
                                logger.log_debug(
                                    f"  Pag {page_idx}: deskew "
                                    f"aplicado ({angle:.2f}°)"
                                )
                except Exception:
                    pass

                # ============================================================
                #  — THRESHOLD CONDICIONAL
                # ============================================================
                if threshold_mode == "binary":
                    # Binarización completa 
                    output_img = cv2.adaptiveThreshold(
                        processed, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                        adaptive_threshold_block,
                        adaptive_threshold_c
                    )
                    kernel     = cv2.getStructuringElement(
                        cv2.MORPH_RECT, (3, 3)
                    )
                    output_img = cv2.morphologyEx(
                        output_img, cv2.MORPH_CLOSE, kernel
                    )
                    output_img = cv2.morphologyEx(
                        output_img, cv2.MORPH_OPEN, kernel
                    )
                    if logger:
                        logger.log_debug(
                            f"  Pag {page_idx}: threshold binary aplicado "
                            f"(block={adaptive_threshold_block}, "
                            f"C={adaptive_threshold_c})"
                        )

                else:
                    # ✅ threshold_mode == "none"
                    # Sin binarización — conserva tonos sepia / grises
                    output_img = processed
                    if logger:
                        logger.log_debug(
                            f"  Pag {page_idx}: threshold OMITIDO "
                            f"— tonos sepia/grises preservados"
                        )

                # ============================================================
                # Guardar página a disco temporal
                # ============================================================
                enhanced_path = workspace.get_page_path(page_idx, "png")
                cv2.imwrite(str(enhanced_path), output_img)
                enhanced_paths.append(enhanced_path)

                if logger:
                    logger.log_debug(
                        f"  Pag {page_idx} guardada: {enhanced_path.name}"
                    )

                # Liberar RAM
                del img_array, img_bgr, gray, processed, output_img, image
                if use_lab:
                    for var in ['img_bgr_eq']:
                        try:
                            del locals()[var]
                        except KeyError:
                            pass
                gc.collect(generation=2)

            except Exception as exc:
                if logger:
                    logger.log_error(
                        f"Error procesando pagina {page_idx}: {exc}"
                    )
                raise

        # ====================================================================
        # Recomponer PDF desde temporales
        # ====================================================================
        if logger:
            logger.log_info(
                f"Recomponiendo PDF desde {len(enhanced_paths)} paginas "
                f"(quality={output_quality}, optimize=True)..."
            )

        try:
            pil_images = []
            for enhanced_path in enhanced_paths:
                pil_img = Image.open(enhanced_path).convert("RGB")
                pil_images.append(pil_img)

            pil_images[0].save(
                str(output_path),
                save_all      = True,
                append_images = pil_images[1:],
                quality       = output_quality,
                optimize      = True,
                dpi           = (dpi, dpi)
            )

            for img in pil_images:
                img.close()
            del pil_images
            gc.collect()

            if logger:
                size_kb = output_path.stat().st_size / 1024
                logger.log_success(
                    f"PDF mejorado guardado: {output_path.name} "
                    f"({size_kb:.1f} KB | quality={output_quality} | "
                    f"threshold={threshold_mode})"
                )

        except Exception as exc:
            if logger:
                logger.log_error(f"Error recomponiendo PDF: {exc}")
            raise

    if logger:
        logger.log_debug("Workspace temporal limpiado automáticamente")

    return str(output_path)


# ============================================================================
# CONVERSIÓN PDF → IMÁGENES
# ============================================================================

def convert_pdf_to_images(
    pdf_path: str,
    dpi:      int = 300,
    logger:   Optional[ProcessingLogger] = None
) -> List[Tuple[np.ndarray, int]]:
    if logger:
        logger.log_info(f"Convirtiendo PDF a imagenes ({dpi} DPI)...")
    images = convert_from_path(pdf_path, dpi=dpi)
    if logger:
        logger.log_success(f"{len(images)} pagina(s) convertida(s)")
    result = []
    for idx, img in enumerate(images):
        img_array = np.array(img)
        img_array, effective_dpi, was_resized = safe_resize_image(
            img_array, original_dpi=dpi, logger=logger
        )
        if was_resized and logger:
            logger.log_info(
                f"  Pagina {idx + 1}: redimensionada "
                f"(DPI efectivo = {effective_dpi})"
            )
        result.append((img_array, idx + 1))
    return result


# ============================================================================
# EASYOCR
# ============================================================================

def initialize_easyocr_reader(
    languages: List[str] = ['es', 'en'],
    use_gpu:   bool      = True,
    logger:    Optional[ProcessingLogger] = None
) -> "easyocr.Reader":
    if not HAS_EASYOCR:
        raise ImportError("EasyOCR requerido: pip install easyocr")
    if logger:
        logger.log_info(
            f"Inicializando EasyOCR (idiomas={languages}, GPU={use_gpu})..."
        )
    try:
        reader = easyocr.Reader(
            languages, gpu=use_gpu,
            model_storage_directory="./easyocr_models"
        )
    except Exception:
        if logger:
            logger.log_warning("GPU no disponible, usando CPU...")
        reader = easyocr.Reader(
            languages, gpu=False,
            model_storage_directory="./easyocr_models"
        )
    if logger:
        logger.log_success("EasyOCR listo")
    return reader


def detect_block_type(
    bbox: Any, text: str, page_height: int, page_width: int
) -> str:
    points      = np.array(bbox)
    y_coords    = points[:, 1]
    y_pos       = float(y_coords.min())
    text_length = len(text)
    words_count = len(text.split())
    if y_pos < page_height * 0.15 and words_count < 15 and text_length < 100:
        return "title"
    if (y_coords.max() - y_coords.min()) < page_height * 0.08 \
            and words_count < 10:
        return "heading"
    if text_length < 50 and y_pos > page_height * 0.8:
        return "caption"
    if "|" in text or "+" in text:
        return "table"
    if text_length > 200:
        return "body"
    return "text"


def extract_structure_easyocr(
    image:    np.ndarray,
    page_num: int,
    reader:   "easyocr.Reader",
    logger:   Optional[ProcessingLogger] = None
) -> PageStructure:
    start_time = datetime.now()
    if logger:
        logger.log_info(f"Analizando pagina {page_num} con EasyOCR...")
    results     = reader.readtext(image, detail=1)
    if logger:
        logger.log_success(
            f"{len(results)} elementos detectados en pagina {page_num}"
        )
    blocks      = []
    full_text   = ""
    page_height, page_width = image.shape[:2]
    for idx, detection in enumerate(results):
        raw_bbox, text, confidence = detection
        points = np.array(raw_bbox)
        x_min  = float(points[:, 0].min())
        y_min  = float(points[:, 1].min())
        x_max  = float(points[:, 0].max())
        y_max  = float(points[:, 1].max())
        bbox_obj   = BoundingBox(
            x=x_min, y=y_min,
            width=x_max - x_min, height=y_max - y_min
        )
        block_type = detect_block_type(
            raw_bbox, text, page_height, page_width
        )
        blocks.append(TextBlock(
            text       = text,
            confidence = float(confidence),
            bbox       = bbox_obj,
            block_type = block_type,
            page_num   = page_num,
            block_id   = f"p{page_num}_b{idx + 1}"
        ))
        full_text += text + " "
    proc_time = (datetime.now() - start_time).total_seconds()
    if logger:
        logger.log_success(
            f"Pagina {page_num}: {len(blocks)} bloques, "
            f"{len(full_text.split())} palabras ({proc_time:.2f}s)"
        )
    return PageStructure(
        page_num        = page_num,
        width           = page_width,
        height          = page_height,
        blocks          = blocks,
        full_text       = full_text.strip(),
        processing_time = proc_time
    )


def extract_all_pages_easyocr(
    image_list: List[Tuple[np.ndarray, int]],
    languages:  List[str] = ['es', 'en'],
    use_gpu:    bool      = True,
    logger:     Optional[ProcessingLogger] = None
) -> List[PageStructure]:
    reader = initialize_easyocr_reader(languages, use_gpu, logger)
    return [
        extract_structure_easyocr(img, page_num, reader, logger)
        for img, page_num in image_list
    ]


# ============================================================================
# ESTADÍSTICAS
# ============================================================================

def generate_ocr_stats(
    page_structures: List[PageStructure],
    logger: Optional[ProcessingLogger] = None
) -> Dict[str, Any]:
    stats = {
        "total_pages":      len(page_structures),
        "total_blocks":     0,
        "total_words":      0,
        "total_characters": 0,
        "avg_confidence":   0.0,
        "block_types":      {},
        "pages":            []
    }
    all_confidences = []
    for ps in page_structures:
        page_confs = []
        page_data  = {
            "page_num":          ps.page_num,
            "total_blocks":      len(ps.blocks),
            "total_words":       len(ps.full_text.split()),
            "total_characters":  len(ps.full_text),
            "avg_confidence":    0.0,
            "block_types_count": {},
            "blocks":            []
        }
        for block in ps.blocks:
            page_confs.append(block.confidence)
            all_confidences.append(block.confidence)
            page_data["blocks"].append({
                "block_id":   block.block_id,
                "type":       block.block_type,
                "text":       block.text[:100],
                "confidence": block.confidence,
                "bbox":       block.bbox.to_dict(),
                "word_count": len(block.text.split())
            })
            stats["block_types"][block.block_type] = \
                stats["block_types"].get(block.block_type, 0) + 1
            page_data["block_types_count"][block.block_type] = \
                page_data["block_types_count"].get(block.block_type, 0) + 1
        if page_confs:
            page_data["avg_confidence"] = sum(page_confs) / len(page_confs)
        stats["total_blocks"]     += len(ps.blocks)
        stats["total_words"]      += len(ps.full_text.split())
        stats["total_characters"] += len(ps.full_text)
        stats["pages"].append(page_data)
    if all_confidences:
        stats["avg_confidence"] = sum(all_confidences) / len(all_confidences)
    if logger:
        logger.log_success(
            f"Estadisticas: {stats['total_blocks']} bloques, "
            f"{stats['total_words']} palabras"
        )
    return stats


# ============================================================================
# PDF SEARCHABLE
# ============================================================================

def create_searchable_pdf(
    pdf_path:        str,
    page_structures: List[PageStructure],
    output_path:     str,
    output_quality:  int  = 72,
    logger: Optional[ProcessingLogger] = None
) -> str:
    if logger:
        logger.log_info(
            f"Creando PDF searchable (quality={output_quality})..."
        )
    images      = convert_from_path(pdf_path, dpi=300)
    rgb_images  = [img.convert("RGB") for img in images]
    output_path_obj = Path(output_path)
    rgb_images[0].save(
        str(output_path_obj),
        save_all      = True,
        append_images = rgb_images[1:],
        quality       = output_quality,
        optimize      = True
    )
    for img in rgb_images:
        img.close()
    del rgb_images
    gc.collect()
    meta_file = output_path_obj.with_suffix(".ocr_meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({
            "ocr_method":  "easyocr",
            "total_pages": len(page_structures),
            "pages": [{
                "page":            ps.page_num,
                "text":            ps.full_text[:1000],
                "blocks_count":    len(ps.blocks),
                "processing_time": ps.processing_time
            } for ps in page_structures]
        }, f, ensure_ascii=False, indent=2)
    if logger:
        size_kb = output_path_obj.stat().st_size / 1024
        logger.log_success(
            f"PDF searchable guardado: {output_path_obj.name} "
            f"({size_kb:.1f} KB | quality={output_quality})"
        )
    return str(output_path_obj)


# ============================================================================
# PROCESAMIENTO DE UN SOLO PDF — RESILIENTE 
# ============================================================================

def process_single_pdf(
    pdf_path:          str,
    base_output_dir:   str                  = "outputs",
    enhance_dpi:       int                  = 400,
    ocr_dpi:           int                  = 300,
    ocr_languages:     Optional[List[str]]  = None,
    denoise:           int                  = 10,
    clahe:             float                = 3.0,
    use_gpu:           bool                 = True,
    use_lab:           bool                 = False,
    output_quality:    int                  = 72,
    threshold_block:   int                  = 21,
    threshold_c:       int                  = 10,
    threshold_mode:    str                  = "none",   
    completados_dir:   Optional[Path]       = None,
    bigsize_dir:       Optional[Path]       = None,
    logger:            Optional[ProcessingLogger] = None
) -> FileResult:
    if logger is None:
        logger = ProcessingLogger()
    if ocr_languages is None:
        ocr_languages = ['es', 'en']

    pdf_path_obj = Path(pdf_path)
    pdf_filename = pdf_path_obj.name
    start_time   = datetime.now()

    file_result = FileResult(
        pdf_name    = pdf_filename,
        pdf_path    = str(pdf_path),
        status      = ProcessStatus.FAILED,
        steps_total = 4
    )

    logger.blank()
    logger.log_debug(f"Liberando memoria antes de procesar '{pdf_filename}'...")
    mem_stats = release_memory(logger)
    logger.log_debug(
        f"GC completado — objetos recogidos: "
        f"{mem_stats.get('gc_objects_collected', 0)}"
    )

    if not pdf_path_obj.exists():
        file_result.error_summary = f"Fichero no encontrado: {pdf_path}"
        file_result.failed_steps.append("file_check: fichero no encontrado")
        file_result.processing_time = (
            datetime.now() - start_time
        ).total_seconds()
        logger.log_error(f"Fichero no encontrado: {pdf_path}")
        return file_result

    output_dir = build_output_dir(base_output_dir, pdf_filename)
    file_result.output_dir = str(output_dir)

    clahe_mode    = "LAB (canal L)" if use_lab else "GRAY clásico"
    denoise_label = f"h={denoise}" if denoise > 0 else "OMITIDO"
    thresh_label  = (
        "DESACTIVADO (tonos preservados)"
        if threshold_mode == "none"
        else f"binary (block={threshold_block}, C={threshold_c})"
    )

    logger.separator()
    logger.log_header(f"PROCESANDO: {pdf_filename}")
    logger.log_info(f"Directorio de salida : {output_dir}")
    logger.log_info(
        f"MAX_IMAGE_PIXELS     : {Image.MAX_IMAGE_PIXELS:,} px"
    )
    logger.log_info(f"Modo CLAHE           : {clahe_mode}")
    logger.log_info(f"Denoise              : {denoise_label}")
    logger.log_info(f"Threshold            : {thresh_label}")   
    logger.log_info(f"Output quality       : {output_quality}")
    logger.log_info(f"Procesamiento v5.7.4 : Workspace temporal por página")
    logger.separator()

    source_pdf      = pdf_path_obj
    page_structures = []

    # ==================================================================
    # PASO 1 — MEJORA DE CALIDAD
    # ==================================================================
    logger.blank()
    logger.log_info(
        f"[PASO 1/4] Mejorando calidad (CLAHE: {'LAB' if use_lab else 'GRAY'}"
        f" | denoise: {denoise_label} | threshold: {threshold_mode})..."
    )
    step_start = datetime.now()

    try:
        enhanced_filename = build_output_filename(
            "01_enhanced_document", pdf_filename, "pdf"
        )
        enhanced_pdf_path = output_dir / enhanced_filename

        enhance_pdf_quality_v5_7(
            str(pdf_path_obj), str(enhanced_pdf_path),
            dpi                      = enhance_dpi,
            denoise_strength         = denoise,
            clahe_clip               = clahe,
            use_lab                  = use_lab,
            output_quality           = output_quality,
            adaptive_threshold_block = threshold_block,
            adaptive_threshold_c     = threshold_c,
            threshold_mode           = threshold_mode,   
            logger                   = logger
        )

        source_pdf = enhanced_pdf_path
        file_result.files_generated.append(str(enhanced_pdf_path))
        file_result.steps_ok += 1
        logger.log_success(
            f"Paso 1 OK "
            f"({(datetime.now() - step_start).total_seconds():.2f}s)"
        )

    except Image.DecompressionBombError as exc:
        return _handle_decompression_bomb(
            pdf_path_obj=pdf_path_obj, file_result=file_result,
            bigsize_dir=bigsize_dir, step="paso1_enhancement",
            exc=exc, start_time=start_time, logger=logger
        )

    except Exception as exc:
        err = f"Enhancement: {type(exc).__name__}: {exc}"
        logger.log_warning(f"Paso 1 fallo (usando PDF original): {err}")
        file_result.failed_steps.append(f"paso1_{err}")
        source_pdf = pdf_path_obj

    # ==================================================================
    # PASO 2 — CONVERSIÓN A IMÁGENES
    # ==================================================================
    logger.blank()
    logger.log_info("[PASO 2/4] Convirtiendo PDF a imagenes...")
    step_start = datetime.now()
    image_list = []

    try:
        image_list = convert_pdf_to_images(
            str(source_pdf), dpi=ocr_dpi, logger=logger
        )
        file_result.steps_ok += 1
        logger.log_success(
            f"Paso 2 OK "
            f"({(datetime.now() - step_start).total_seconds():.2f}s)"
        )

    except Image.DecompressionBombError as exc:
        return _handle_decompression_bomb(
            pdf_path_obj=pdf_path_obj, file_result=file_result,
            bigsize_dir=bigsize_dir, step="paso2_image_conversion",
            exc=exc, start_time=start_time, logger=logger
        )

    except Exception as exc:
        err = f"Conversion imagenes: {type(exc).__name__}: {exc}"
        logger.log_error(f"Paso 2 CRITICO: {err}")
        logger.log_debug(traceback.format_exc())
        file_result.failed_steps.append(f"paso2_{err}")
        file_result.error_summary   = err
        file_result.status          = ProcessStatus.FAILED
        file_result.processing_time = (
            datetime.now() - start_time
        ).total_seconds()
        return file_result

    # ==================================================================
    # PASO 3 — OCR + LAYOUT
    # ==================================================================
    logger.blank()
    logger.log_info("[PASO 3/4] Extrayendo texto con EasyOCR + Layout...")
    step_start = datetime.now()

    try:
        page_structures = extract_all_pages_easyocr(
            image_list, languages=ocr_languages,
            use_gpu=use_gpu, logger=logger
        )
        file_result.steps_ok += 1
        logger.log_success(
            f"Paso 3 OK "
            f"({(datetime.now() - step_start).total_seconds():.2f}s)"
        )

    except Exception as exc:
        err = f"OCR: {type(exc).__name__}: {exc}"
        logger.log_error(f"Paso 3 CRITICO: {err}")
        logger.log_debug(traceback.format_exc())
        file_result.failed_steps.append(f"paso3_{err}")
        file_result.error_summary   = err
        file_result.status          = ProcessStatus.FAILED
        file_result.processing_time = (
            datetime.now() - start_time
        ).total_seconds()
        return file_result

    # ==================================================================
    # PASO 4 — ARCHIVOS DE SALIDA
    # ==================================================================
    logger.blank()
    logger.log_info("[PASO 4/4] Generando archivos de salida...")
    outputs_ok = 0

    # 4.1 — ocr_stats
    try:
        stats_filename = build_output_filename(
            "02_ocr_stats", pdf_filename, "json"
        )
        stats_file     = output_dir / stats_filename
        logger.log_info(f"  Generando {stats_filename}...")
        ocr_stats_data = generate_ocr_stats(page_structures, logger)
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(ocr_stats_data, f, ensure_ascii=False, indent=2)
        file_result.files_generated.append(str(stats_file))
        file_result.statistics = {
            "total_pages":      ocr_stats_data["total_pages"],
            "total_blocks":     ocr_stats_data["total_blocks"],
            "total_words":      ocr_stats_data["total_words"],
            "total_characters": ocr_stats_data["total_characters"],
            "avg_confidence":   f"{ocr_stats_data['avg_confidence']:.2%}",
            "block_types":      ocr_stats_data["block_types"]
        }
        outputs_ok += 1
        logger.log_success(f"  Guardado: {stats_filename}")
    except Exception as exc:
        err = f"ocr_stats: {type(exc).__name__}: {exc}"
        logger.log_warning(f"  Fallo stats: {err}")
        file_result.failed_steps.append(f"paso4_{err}")

    # 4.2 — ocr_searchable
    try:
        searchable_filename = build_output_filename(
            "03_ocr_searchable", pdf_filename, "pdf"
        )
        searchable_file = output_dir / searchable_filename
        logger.log_info(f"  Generando {searchable_filename}...")
        create_searchable_pdf(
            str(source_pdf), page_structures,
            str(searchable_file),
            output_quality=output_quality, logger=logger
        )
        file_result.files_generated.append(str(searchable_file))
        outputs_ok += 1
        logger.log_success(f"  Guardado: {searchable_filename}")
    except Exception as exc:
        err = f"ocr_searchable: {type(exc).__name__}: {exc}"
        logger.log_warning(f"  Fallo searchable PDF: {err}")
        file_result.failed_steps.append(f"paso4_{err}")

    # 4.3 — complete_structure
    try:
        structure_filename = build_output_filename(
            "04_complete_structure", pdf_filename, "json"
        )
        structure_file = output_dir / structure_filename
        logger.log_info(f"  Generando {structure_filename}...")
        with open(structure_file, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "pdf_path":        str(pdf_path),
                    "pdf_name":        pdf_filename,
                    "processing_date": datetime.now().isoformat(),
                    "languages":       ocr_languages,
                    "clahe_mode":      "LAB" if use_lab else "GRAY",
                    "output_quality":  output_quality,
                    "threshold_mode":  threshold_mode,      
                    "threshold_block": threshold_block,
                    "threshold_c":     threshold_c
                },
                "summary": {
                    "total_pages":  len(page_structures),
                    "total_blocks": sum(
                        len(ps.blocks) for ps in page_structures
                    ),
                    "total_words":  sum(
                        len(ps.full_text.split()) for ps in page_structures
                    )
                },
                "pages": [ps.to_dict() for ps in page_structures]
            }, f, ensure_ascii=False, indent=2)
        file_result.files_generated.append(str(structure_file))
        outputs_ok += 1
        logger.log_success(f"  Guardado: {structure_filename}")
    except Exception as exc:
        err = f"complete_structure: {type(exc).__name__}: {exc}"
        logger.log_warning(f"  Fallo estructura: {err}")
        file_result.failed_steps.append(f"paso4_{err}")

    file_result.steps_ok       += 1
    file_result.processing_time = (datetime.now() - start_time).total_seconds()

    if not file_result.failed_steps:
        file_result.status = ProcessStatus.SUCCESS
    elif outputs_ok > 0:
        file_result.status = ProcessStatus.PARTIAL
    else:
        file_result.status = ProcessStatus.FAILED

    if file_result.failed_steps:
        file_result.error_summary = "; ".join(file_result.failed_steps[:2])

    # 4.4 — results JSON
    try:
        results_filename = build_output_filename(
            "05_results", pdf_filename, "json"
        )
        results_file = output_dir / results_filename
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(file_result.to_dict(), f, ensure_ascii=False, indent=2)
        file_result.files_generated.append(str(results_file))
        logger.log_success(f"  Guardado: {results_filename}")
    except Exception as exc:
        logger.log_warning(f"  Fallo results JSON: {exc}")

    # Mover a completados si SUCCESS
    if file_result.status == ProcessStatus.SUCCESS \
            and completados_dir is not None:
        logger.blank()
        logger.log_info(
            f"Proceso exitoso — moviendo '{pdf_filename}' a 'completados'..."
        )
        moved, completed_path = move_file_to_folder(
            str(pdf_path_obj), completados_dir,
            reason="proceso completado", logger=logger
        )
        file_result.moved_to_completed = moved
        file_result.completed_path     = completed_path
        if not moved:
            logger.log_warning(
                f"'{pdf_filename}' NO pudo moverse a completados."
            )
    elif file_result.status != ProcessStatus.SUCCESS:
        logger.log_info(
            f"Fichero NO movido a completados "
            f"(status: {file_result.status.value})"
        )

    logger.blank()
    logger.separator("-")
    if file_result.status == ProcessStatus.SUCCESS:
        logger.log_success(
            f"COMPLETADO [SUCCESS]: {pdf_filename} "
            f"({file_result.processing_time:.2f}s)"
        )
    elif file_result.status == ProcessStatus.PARTIAL:
        logger.log_warning(
            f"PARCIAL [PARTIAL]: {pdf_filename} "
            f"({file_result.processing_time:.2f}s)"
        )
    else:
        logger.log_error(
            f"FALLIDO [FAILED]: {pdf_filename} "
            f"({file_result.processing_time:.2f}s)"
        )
    if file_result.moved_to_completed:
        logger.log_success(
            f"Movido a completados: "
            f"{Path(file_result.completed_path).name}"
        )
    logger.separator("-")
    logger.blank()

    return file_result


# ============================================================================
# HANDLER DecompressionBombError
# ============================================================================

def _handle_decompression_bomb(
    pdf_path_obj: Path,
    file_result:  FileResult,
    bigsize_dir:  Optional[Path],
    step:         str,
    exc:          Exception,
    start_time:   datetime,
    logger:       ProcessingLogger
) -> FileResult:
    err_msg = (
        f"DecompressionBombError en {step}: imagen demasiado grande "
        f"— fichero movido a bigsizeDocuments"
    )
    logger.blank()
    logger.log_bigsize(f"BIGSIZE DETECTADO en '{pdf_path_obj.name}': {exc}")
    logger.log_bigsize(
        f"Supera el limite ({ABSOLUTE_PIXEL_LIMIT:,} px). "
        f"Se omite y mueve a 'bigsizeDocuments'."
    )
    file_result.status        = ProcessStatus.BIGSIZE
    file_result.error_summary = err_msg
    file_result.failed_steps.append(
        f"{step}: DecompressionBombError: {exc}"
    )
    file_result.processing_time = (
        datetime.now() - start_time
    ).total_seconds()
    if bigsize_dir is not None:
        moved, bigsize_path = move_file_to_folder(
            str(pdf_path_obj), bigsize_dir,
            reason="DecompressionBombError", logger=logger
        )
        file_result.moved_to_bigsize = moved
        file_result.bigsize_path     = bigsize_path
        if not moved:
            logger.log_warning(
                f"'{pdf_path_obj.name}' NO pudo moverse a bigsizeDocuments."
            )
    else:
        logger.log_warning(
            "Directorio bigsizeDocuments no configurado."
        )
    gc.collect()
    logger.separator("-")
    logger.log_bigsize(
        f"BIGSIZE [OMITIDO]: {pdf_path_obj.name} "
        f"({file_result.processing_time:.2f}s)"
    )
    if file_result.moved_to_bigsize:
        logger.log_bigsize(
            f"Movido a bigsizeDocuments: "
            f"{Path(file_result.bigsize_path).name}"
        )
    logger.separator("-")
    logger.blank()
    return file_result


# ============================================================================
# PROCESAMIENTO EN LOTE
# ============================================================================

def process_batch(
    pdf_paths:        List[str],
    base_output_dir:  str                  = "outputs",
    enhance_dpi:      int                  = 400,
    ocr_dpi:          int                  = 300,
    ocr_languages:    Optional[List[str]]  = None,
    denoise:          int                  = 10,
    clahe:            float                = 3.0,
    use_gpu:          bool                 = True,
    use_lab:          bool                 = False,
    output_quality:   int                  = 72,
    threshold_block:  int                  = 21,
    threshold_c:      int                  = 10,
    threshold_mode:   str                  = "none",   
    logger:           Optional[ProcessingLogger] = None
) -> Dict[str, Any]:
    if logger is None:
        logger = ProcessingLogger()

    batch_start   = datetime.now()
    file_results: List[FileResult] = []

    completados_dir = ensure_special_folder(base_output_dir, "completados")
    bigsize_dir     = ensure_special_folder(base_output_dir, "bigsizeDocuments")

    clahe_mode_label = "LAB (canal L)" if use_lab else "GRAY clásico"
    denoise_label    = f"h={denoise}" if denoise > 0 else "OMITIDO"
    thresh_label     = (
        "DESACTIVADO (tonos preservados)"
        if threshold_mode == "none"
        else f"binary (block={threshold_block}, C={threshold_c})"
    )

    logger.separator("=")
    logger.log_header("OCR + LAYOUT ANALYSIS v5.7.4 — BATCH RESILIENTE")
    logger.log_header(f"Ficheros a procesar  : {len(pdf_paths)}")
    logger.log_header(f"Salida base          : {base_output_dir}")
    logger.log_header(f"Modo CLAHE           : {clahe_mode_label}")
    logger.log_header(f"Denoise              : {denoise_label}")
    logger.log_header(f"Threshold            : {thresh_label}")   
    logger.log_header(f"Output quality       : {output_quality}")
    logger.log_header(f"MAX_IMAGE_PIXELS     : {Image.MAX_IMAGE_PIXELS:,} px")
    logger.log_header(f"OCR minimo garantizado: {OCR_MIN_DPI} DPI")
    logger.log_header(f"Procesamiento        : v5.7.4 (Workspace temporal)")
    logger.separator("=")

    for idx, pdf_path in enumerate(pdf_paths, 1):
        pdf_name = Path(pdf_path).name
        logger.blank()
        logger.log_info(f"[{idx}/{len(pdf_paths)}] Iniciando: {pdf_name}")
        try:
            result = process_single_pdf(
                pdf_path        = pdf_path,
                base_output_dir = base_output_dir,
                enhance_dpi     = enhance_dpi,
                ocr_dpi         = ocr_dpi,
                ocr_languages   = ocr_languages,
                denoise         = denoise,
                clahe           = clahe,
                use_gpu         = use_gpu,
                use_lab         = use_lab,
                output_quality  = output_quality,
                threshold_block = threshold_block,
                threshold_c     = threshold_c,
                threshold_mode  = threshold_mode,   
                completados_dir = completados_dir,
                bigsize_dir     = bigsize_dir,
                logger          = logger
            )
            file_results.append(result)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            logger.log_error(f"Error inesperado en '{pdf_name}': {err}")
            logger.log_debug(traceback.format_exc())
            gc.collect()
            file_results.append(FileResult(
                pdf_name      = pdf_name,
                pdf_path      = pdf_path,
                status        = ProcessStatus.FAILED,
                error_summary = err,
                failed_steps  = [f"unexpected_error: {err}"]
            ))

    total_success = sum(
        1 for r in file_results if r.status == ProcessStatus.SUCCESS
    )
    total_partial = sum(
        1 for r in file_results if r.status == ProcessStatus.PARTIAL
    )
    total_failed  = sum(
        1 for r in file_results
        if r.status in (ProcessStatus.FAILED, ProcessStatus.SKIPPED)
    )
    total_bigsize = sum(
        1 for r in file_results if r.status == ProcessStatus.BIGSIZE
    )
    total_moved   = sum(1 for r in file_results if r.moved_to_completed)
    batch_time    = (datetime.now() - batch_start).total_seconds()

    _print_final_status_log(
        file_results=file_results, batch_time=batch_time,
        total_success=total_success, total_partial=total_partial,
        total_failed=total_failed, total_bigsize=total_bigsize,
        total_moved=total_moved, completados_dir=completados_dir,
        bigsize_dir=bigsize_dir, logger=logger
    )

    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    batch_summary = {
        "batch_date":      datetime.now().isoformat(),
        "total_files":     len(pdf_paths),
        "total_success":   total_success,
        "total_partial":   total_partial,
        "total_failed":    total_failed,
        "total_bigsize":   total_bigsize,
        "total_moved":     total_moved,
        "completados_dir": str(completados_dir),
        "bigsize_dir":     str(bigsize_dir),
        "total_time_sec":  round(batch_time, 2),
        "version":         "5.7.4",
        "clahe_mode":      "LAB" if use_lab else "GRAY",
        "output_quality":  output_quality,
        "threshold_mode":  threshold_mode,     
        "threshold_block": threshold_block,
        "threshold_c":     threshold_c,
        "documents":       [r.to_dict() for r in file_results]
    }

    batch_summary_file = Path(base_output_dir) / "batch_summary.json"
    with open(batch_summary_file, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2)

    logger.log_info(f"Resumen batch guardado en: {batch_summary_file}")
    logger.blank()
    return batch_summary


# ============================================================================
# LOG FINAL
# ============================================================================

def _print_final_status_log(
    file_results:    List[FileResult],
    batch_time:      float,
    total_success:   int,
    total_partial:   int,
    total_failed:    int,
    total_bigsize:   int,
    total_moved:     int,
    completados_dir: Path,
    bigsize_dir:     Path,
    logger:          ProcessingLogger
):
    STATUS_COLORS = {
        ProcessStatus.SUCCESS: "\033[32m",
        ProcessStatus.PARTIAL: "\033[33m",
        ProcessStatus.FAILED:  "\033[31m",
        ProcessStatus.SKIPPED: "\033[36m",
        ProcessStatus.BIGSIZE: "\033[95m",
    }
    STATUS_ICONS = {
        ProcessStatus.SUCCESS: "[OK]     ",
        ProcessStatus.PARTIAL: "[PARCIAL]",
        ProcessStatus.FAILED:  "[FALLO]  ",
        ProcessStatus.SKIPPED: "[SKIP]   ",
        ProcessStatus.BIGSIZE: "[BIGSIZE]",
    }
    DEST_ICONS = {
        "completed": "\033[32m[COMPLETADOS]\033[0m",
        "bigsize":   "\033[95m[BIGSIZE]    \033[0m",
        "origin":    "\033[90m[ORIGEN]     \033[0m",
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    max_name_len = max((len(r.pdf_name) for r in file_results), default=30)
    max_name_len = max(max_name_len, 32)

    col_w = {
        "idx": 4, "name": max_name_len, "status": 11,
        "dest": 15, "time": 9, "pages": 7, "words": 9, "error": 36
    }

    header_line = (
        f"{'#':>{col_w['idx']}}  "
        f"{'FICHERO PDF':<{col_w['name']}}  "
        f"{'STATUS':<{col_w['status']}}  "
        f"{'DESTINO':<{col_w['dest']}}  "
        f"{'TIEMPO':>{col_w['time']}}  "
        f"{'PAGS':>{col_w['pages']}}  "
        f"{'PALABRAS':>{col_w['words']}}  "
        f"{'ERROR / DETALLE':<{col_w['error']}}"
    )
    separator = "-" * len(header_line)

    logger.blank()
    logger.separator("=")
    print(f"{BOLD}  RESUMEN FINAL — OCR v5.7.4{RESET}")
    logger.separator("=")
    logger.blank()
    print(f"{BOLD}{header_line}{RESET}")
    print(separator)

    for idx, r in enumerate(file_results, 1):
        s_color = STATUS_COLORS.get(r.status, "")
        s_icon  = STATUS_ICONS.get(r.status, r.status.value)
        if r.moved_to_completed:
            dest_icon = DEST_ICONS["completed"]
        elif r.moved_to_bigsize:
            dest_icon = DEST_ICONS["bigsize"]
        else:
            dest_icon = DEST_ICONS["origin"]
        pages  = r.statistics.get("total_pages", "-") if r.statistics else "-"
        words  = r.statistics.get("total_words", "-") if r.statistics else "-"
        time_s = f"{r.processing_time:.1f}s" if r.processing_time > 0 else "-"
        error_str = ""
        if r.error_summary:
            error_str = (
                (r.error_summary[:col_w["error"] - 3] + "...")
                if len(r.error_summary) > col_w["error"]
                else r.error_summary
            )
        name_display = r.pdf_name
        if len(name_display) > col_w["name"]:
            name_display = name_display[:col_w["name"] - 3] + "..."
        print(
            f"{BOLD}{idx:>{col_w['idx']}}{RESET}  "
            f"{name_display:<{col_w['name']}}  "
            f"{s_color}{s_icon:<{col_w['status']}}{RESET}  "
            f"{dest_icon}  "
            f"{time_s:>{col_w['time']}}  "
            f"{str(pages):>{col_w['pages']}}  "
            f"{str(words):>{col_w['words']}}  "
            f"{error_str:<{col_w['error']}}"
        )
        logger._logger.info(
            f"[{idx}] {r.pdf_name} | {r.status.value} | "
            f"completados={r.moved_to_completed} | "
            f"bigsize={r.moved_to_bigsize} | "
            f"{time_s} | pages={pages} | words={words} | "
            f"error={r.error_summary or 'none'}"
        )

    print(separator)
    logger.blank()

    total = len(file_results)
    def pct(n): return f"{n / total * 100:.0f}%" if total else "0%"

    print(
        f"  {BOLD}TOTALES:{RESET}  "
        f"\033[32m{total_success} exitosos ({pct(total_success)}){RESET}  |  "
        f"\033[33m{total_partial} parciales ({pct(total_partial)}){RESET}  |  "
        f"\033[31m{total_failed} fallidos ({pct(total_failed)}){RESET}  |  "
        f"\033[95m{total_bigsize} bigsize ({pct(total_bigsize)}){RESET}  |  "
        f"Total: {total}"
    )
    print(
        f"  \033[32m{total_moved} fichero(s) → completados{RESET}  "
        f"({completados_dir})"
    )
    bigs_moved = sum(1 for r in file_results if r.moved_to_bigsize)
    print(
        f"  \033[95m{bigs_moved} fichero(s) → bigsizeDocuments{RESET}  "
        f"({bigsize_dir})"
    )
    print(
        f"  Tiempo total: {BOLD}{batch_time:.2f}s{RESET} "
        f"({batch_time / 60:.1f} min)"
    )

    logger.blank()
    logger.separator("=")

    errored = [
        r for r in file_results
        if r.failed_steps and r.status != ProcessStatus.BIGSIZE
    ]
    if errored:
        print(f"\n{BOLD}  DETALLE DE ERRORES:{RESET}")
        logger.separator("-")
        for r in errored:
            color = STATUS_COLORS.get(r.status, "")
            print(f"\n  {color}[{r.status.value}]{RESET}  {r.pdf_name}")
            for step_err in r.failed_steps:
                print(f"           - {step_err}")
        logger.separator("-")
        logger.blank()

    bigsize_list = [r for r in file_results if r.status == ProcessStatus.BIGSIZE]
    if bigsize_list:
        print(f"\n{BOLD}  FICHEROS MOVIDOS A bigsizeDocuments:{RESET}")
        logger.separator("-")
        for r in bigsize_list:
            dest_name = (
                Path(r.bigsize_path).name if r.bigsize_path else r.pdf_name
            )
            moved_str = (
                f"\033[95m[MOVIDO]{RESET}" if r.moved_to_bigsize
                else f"\033[31m[SIN MOVER]{RESET}"
            )
            print(
                f"  {moved_str}  {dest_name}  "
                f"\033[90m(DecompressionBombError){RESET}"
            )
        logger.separator("-")
        logger.blank()

    moved_list = [r for r in file_results if r.moved_to_completed]
    if moved_list:
        print(f"\n{BOLD}  FICHEROS EN completados:{RESET}")
        logger.separator("-")
        for r in moved_list:
            dest_name = (
                Path(r.completed_path).name
                if r.completed_path else r.pdf_name
            )
            print(f"  \033[32m[OK]{RESET}  {dest_name}")
        logger.separator("-")
        logger.blank()


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "OCR + LAYOUT ANALYSIS v5.7.4 — Resiliente + completados "
            "+ bigsizeDocuments + Workspace Temporal + CLAHE LAB "
            "+ Output Quality + Threshold CLI + Threshold Mode"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Documentos sepia / históricos (fondo preservado):
  python3 enviarOCR.py doc.pdf -o procesados/ \\
    --use-lab --threshold-mode none

  # Documentos modernos (fondo blanco, binarización):
  python3 enviarOCR.py doc.pdf -o procesados/ \\
    --threshold-mode binary --threshold-block 15 --threshold-c 8

  # Comando recomendado completo para sepia:
  python3 enviarOCR.py repositorioPDFs/doc.pdf \\
    -o procesados/ \\
    --enhance-dpi 400 \\
    --ocr-dpi 300 \\
    --ocr-languages es \\
    --clahe 2.0 \\
    --denoise 5 \\
    --use-lab \\
    --threshold-mode none \\
    --output-quality 90 \\
    --no-gpu

Valores --threshold-mode:
  none   → SIN binarización. Conserva tonos sepia/grises/color.
           Recomendado para documentos históricos, fondos coloreados.  ← DEFAULT
  binary → Binarización adaptativa + morfología. Para docs modernos
           con fondo blanco puro y texto negro definido.

Valores recomendados para --threshold-block (solo con --threshold-mode binary):
  11 → trazo fino, docs alta resolución
  15 → texto impreso limpio
  21 → sepia / documentos degradados
  31 → iluminación muy irregular

Tamaño PDFs (--output-quality):
  95 → ~150 MB  85 → ~60 MB  72 → ~18 MB ←OCR  60 → ~10 MB

Estructura generada:
  outputs/
  +-- completados/
  +-- bigsizeDocuments/
  +-- output_<nombre>/
  |   +-- 01_enhanced_document_<nombre>.pdf
  |   +-- 02_ocr_stats_<nombre>.json
  |   +-- 03_ocr_searchable_<nombre>.pdf
  |   +-- 04_complete_structure_<nombre>.json
  |   +-- 05_results_<nombre>.json
  +-- batch_summary.json
  +-- logs/

        """
    )

    parser.add_argument(
        "sources", nargs="+",
        help="Ruta(s): carpeta, patrón glob o archivo PDF"
    )
    parser.add_argument(
        "-o", "--output", default="outputs",
        help="Directorio base de salida (default: outputs)"
    )
    parser.add_argument(
        "--enhance-dpi", type=int, default=400,
        help="DPI para mejora de imagen (default: 400)"
    )
    parser.add_argument(
        "--ocr-dpi", type=int, default=300,
        help="DPI mínimo para OCR (default: 300)"
    )
    parser.add_argument(
        "--ocr-languages", nargs="+", default=["es", "en"],
        help="Idiomas EasyOCR (default: es en)"
    )
    parser.add_argument(
        "--denoise", type=int, default=10,
        help="Fuerza de denoise (default: 10). 0 = omitir."
    )
    parser.add_argument(
        "--clahe", type=float, default=3.0,
        help="CLAHE clip limit (default: 3.0). 0 = sin CLAHE."
    )
    parser.add_argument(
        "--use-gpu", action="store_true", default=True,
        help="Usar GPU (default: True)"
    )
    parser.add_argument(
        "--no-gpu", dest="use_gpu", action="store_false",
        help="Forzar CPU"
    )
    parser.add_argument(
        "--use-lab", dest="use_lab", action="store_true", default=False,
        help=(
            "CLAHE en espacio LAB (canal L). "
            "Recomendado para fondos de color (sepia, amarillo)."
        )
    )
    parser.add_argument(
        "--output-quality", dest="output_quality", type=int, default=72,
        help="Calidad JPEG PDFs generados (1-95). Default: 72."
    )
    parser.add_argument(
        "--threshold-block", dest="threshold_block", type=int, default=21,
        help=(
            "Bloque umbral adaptativo (impar: 11,15,21,31). "
            "Solo activo con --threshold-mode binary. Default: 21."
        )
    )
    parser.add_argument(
        "--threshold-c", dest="threshold_c", type=int, default=10,
        help=(
            "Constante C del umbral adaptativo (rango: 5-15). "
            "Solo activo con --threshold-mode binary. Default: 10."
        )
    )
    # 
    parser.add_argument(
        "--threshold-mode", dest="threshold_mode",
        choices=THRESHOLD_MODES,
        default="none",
        help=(
            "Modo de binarización: "
            "'none' = SIN threshold, conserva sepia/tonos (DEFAULT); "
            "'binary' = threshold adaptativo + morfología (docs modernos)."
        )
    )

    args        = parser.parse_args()
    valid_paths = expand_pdf_sources(args.sources)

    if not valid_paths:
        print("[ERROR] No se encontraron PDFs válidos.")
        sys.exit(1)

    logger = ProcessingLogger()

    if len(valid_paths) == 1:
        completados_dir = ensure_special_folder(args.output, "completados")
        bigsize_dir     = ensure_special_folder(args.output, "bigsizeDocuments")

        thresh_label = (
            "DESACTIVADO (tonos preservados)"
            if args.threshold_mode == "none"
            else f"binary (block={args.threshold_block}, C={args.threshold_c})"
        )

        logger.log_info(f"Carpeta completados     : {completados_dir}")
        logger.log_info(f"Carpeta bigsizeDocuments: {bigsize_dir}")
        logger.log_info(
            f"Modo CLAHE              : "
            f"{'LAB (canal L)' if args.use_lab else 'GRAY clásico'}"
        )
        logger.log_info(
            f"Denoise                 : "
            f"{'h=' + str(args.denoise) if args.denoise > 0 else 'OMITIDO'}"
        )
        logger.log_info(f"Threshold               : {thresh_label}")
        logger.log_info(f"Output quality          : {args.output_quality}")

        result = process_single_pdf(
            pdf_path        = valid_paths[0],
            base_output_dir = args.output,
            enhance_dpi     = args.enhance_dpi,
            ocr_dpi         = args.ocr_dpi,
            ocr_languages   = args.ocr_languages,
            denoise         = args.denoise,
            clahe           = args.clahe,
            use_gpu         = args.use_gpu,
            use_lab         = args.use_lab,
            output_quality  = args.output_quality,
            threshold_block = args.threshold_block,
            threshold_c     = args.threshold_c,
            threshold_mode  = args.threshold_mode,   
            completados_dir = completados_dir,
            bigsize_dir     = bigsize_dir,
            logger          = logger
        )

        _print_final_status_log(
            file_results    = [result],
            batch_time      = result.processing_time,
            total_success   = 1 if result.status == ProcessStatus.SUCCESS  else 0,
            total_partial   = 1 if result.status == ProcessStatus.PARTIAL  else 0,
            total_failed    = 1 if result.status == ProcessStatus.FAILED   else 0,
            total_bigsize   = 1 if result.status == ProcessStatus.BIGSIZE  else 0,
            total_moved     = 1 if result.moved_to_completed               else 0,
            completados_dir = completados_dir,
            bigsize_dir     = bigsize_dir,
            logger          = logger
        )

        sys.exit(0 if result.status not in (
            ProcessStatus.FAILED, ProcessStatus.BIGSIZE
        ) else 1)

    else:
        batch = process_batch(
            pdf_paths       = valid_paths,
            base_output_dir = args.output,
            enhance_dpi     = args.enhance_dpi,
            ocr_dpi         = args.ocr_dpi,
            ocr_languages   = args.ocr_languages,
            denoise         = args.denoise,
            clahe           = args.clahe,
            use_gpu         = args.use_gpu,
            use_lab         = args.use_lab,
            output_quality  = args.output_quality,
            threshold_block = args.threshold_block,
            threshold_c     = args.threshold_c,
            threshold_mode  = args.threshold_mode,   
            logger          = logger
        )

        sys.exit(0 if batch["total_failed"] == 0 and
                     batch["total_bigsize"] == 0 else 1)


if __name__ == "__main__":
    main()