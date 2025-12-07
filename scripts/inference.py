#!/usr/bin/env python3
"""
=============================================================================
Script de Inferencia - Detección de Cajas VR con modelo entrenado
=============================================================================

Este script utiliza un modelo YOLOv8/v11 entrenado para detectar cajas VR
en nuevas imágenes, videos o streams.

Uso:
    # Imagen individual
    python scripts/inference.py --source imagen.jpg

    # Directorio de imágenes
    python scripts/inference.py --source carpeta/imagenes/

    # Video
    python scripts/inference.py --source video.mp4

    # Webcam
    python scripts/inference.py --source 0

    # Con modelo específico
    python scripts/inference.py --source imagen.jpg --model models/vr_boxes_best_20241201.pt

Autor: [Tu nombre]
Fecha: Diciembre 2024
=============================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURACIÓN INICIAL
# =============================================================================

# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def find_best_model() -> Path:
    """
    Busca el mejor modelo disponible en el proyecto.

    Prioridad de búsqueda:
    1. Modelo más reciente en models/ (versionado con timestamp)
    2. best.pt en el último experimento de runs/train/
    3. Modelo base yolov8n.pt (sin entrenar)

    Returns:
        Path: Ruta al mejor modelo encontrado

    Raises:
        FileNotFoundError: Si no se encuentra ningún modelo
    """
    # 1. Buscar en models/ (modelos versionados)
    models_dir = PROJECT_ROOT / 'models'
    if models_dir.exists():
        versioned_models = sorted(models_dir.glob('vr_boxes_*.pt'), reverse=True)
        if versioned_models:
            logger.info(f"Encontrado modelo versionado: {versioned_models[0].name}")
            return versioned_models[0]

    # 2. Buscar en runs/train/ (último experimento)
    runs_dir = PROJECT_ROOT / 'runs' / 'train'
    if runs_dir.exists():
        experiments = sorted(runs_dir.glob('*/weights/best.pt'), reverse=True)
        if experiments:
            logger.info(f"Encontrado modelo de entrenamiento: {experiments[0]}")
            return experiments[0]

    # 3. No se encontró modelo entrenado
    raise FileNotFoundError(
        "No se encontró ningún modelo entrenado.\n"
        "Ejecuta primero: python scripts/train.py"
    )


def run_inference(args):
    """
    Ejecuta la inferencia en las imágenes/video especificados.

    Esta función:
    1. Carga el modelo entrenado
    2. Procesa las fuentes de entrada
    3. Genera predicciones con bounding boxes
    4. Guarda y/o muestra los resultados

    Args:
        args: Argumentos parseados de la línea de comandos
    """
    from ultralytics import YOLO

    # =========================================================================
    # PASO 1: CARGAR MODELO
    # =========================================================================
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(f"Modelo no encontrado: {args.model}")
            sys.exit(1)
    else:
        try:
            model_path = find_best_model()
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    logger.info(f"Cargando modelo: {model_path}")
    model = YOLO(str(model_path))

    # Mostrar información del modelo
    logger.info(f"Clases del modelo: {model.names}")

    # =========================================================================
    # PASO 2: VERIFICAR FUENTE
    # =========================================================================
    source = args.source

    # Verificar si es un archivo/directorio existente
    if source.isdigit():
        # Es una webcam (0, 1, etc.)
        source = int(source)
        logger.info(f"Fuente: Webcam {source}")
    elif Path(source).exists():
        source_path = Path(source)
        if source_path.is_file():
            logger.info(f"Fuente: Archivo {source_path.name}")
        else:
            num_files = len(list(source_path.glob('*')))
            logger.info(f"Fuente: Directorio con {num_files} archivos")
    else:
        logger.error(f"Fuente no encontrada: {source}")
        sys.exit(1)

    # =========================================================================
    # PASO 3: CONFIGURAR SALIDA
    # =========================================================================
    # Crear directorio de salida con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / 'runs' / 'inference' / f'predict_{timestamp}'

    logger.info(f"Resultados se guardarán en: {output_dir}")

    # =========================================================================
    # PASO 4: EJECUTAR INFERENCIA
    # =========================================================================
    logger.info("=" * 60)
    logger.info("EJECUTANDO INFERENCIA")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Parámetros de predicción
    # -------------------------------------------------------------------------
    # conf: Umbral de confianza mínimo para detecciones
    #       - 0.25: Más detecciones, más falsos positivos
    #       - 0.5: Balance
    #       - 0.7+: Menos detecciones, más precisas
    #
    # iou: Umbral de IoU para NMS (Non-Maximum Suppression)
    #       - 0.45: Valor por defecto, funciona bien
    #       - Reducir si hay muchas detecciones duplicadas
    #
    # max_det: Máximo número de detecciones por imagen
    #       - Ajustar según cuántas cajas esperas en una imagen
    # -------------------------------------------------------------------------

    results = model.predict(
        source=source,
        conf=args.conf,                    # Umbral de confianza
        iou=args.iou,                      # Umbral de IoU para NMS
        max_det=args.max_det,              # Máximo detecciones por imagen
        save=args.save,                    # Guardar imágenes con detecciones
        save_txt=args.save_txt,            # Guardar etiquetas en formato YOLO
        save_conf=args.save_conf,          # Incluir confianza en etiquetas
        save_crop=args.save_crop,          # Guardar recortes de detecciones
        show=args.show,                    # Mostrar resultados en ventana
        project=str(PROJECT_ROOT / 'runs' / 'inference'),
        name=f'predict_{timestamp}',
        exist_ok=True,
        verbose=args.verbose,
        device=args.device,
        stream=True,                       # Modo streaming para eficiencia
    )

    # =========================================================================
    # PASO 5: PROCESAR Y MOSTRAR RESULTADOS
    # =========================================================================
    total_detections = 0
    processed_files = 0

    for result in results:
        processed_files += 1

        # Obtener información de las detecciones
        boxes = result.boxes
        num_detections = len(boxes)
        total_detections += num_detections

        if num_detections > 0:
            logger.info(f"\n📁 {result.path}")
            logger.info(f"   Detecciones: {num_detections}")

            # Mostrar detalles de cada detección
            for i, box in enumerate(boxes):
                # Obtener clase, confianza y coordenadas
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]

                # Coordenadas del bounding box (x1, y1, x2, y2)
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(c) for c in coords]

                logger.info(f"   [{i+1}] {class_name}: {conf:.2%} @ ({x1}, {y1}, {x2}, {y2})")

    # =========================================================================
    # PASO 6: RESUMEN FINAL
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN DE INFERENCIA")
    logger.info("=" * 60)
    logger.info(f"Archivos procesados: {processed_files}")
    logger.info(f"Total de detecciones: {total_detections}")
    logger.info(f"Promedio por archivo: {total_detections/max(processed_files, 1):.1f}")

    if args.save:
        logger.info(f"\nResultados guardados en:")
        logger.info(f"  {output_dir}")

    logger.info("=" * 60)


def export_model(args):
    """
    Exporta el modelo a diferentes formatos para producción.

    Formatos disponibles:
    - ONNX: Para inferencia en diversos frameworks
    - TensorRT: Optimizado para GPUs NVIDIA
    - CoreML: Para dispositivos Apple
    - TFLite: Para dispositivos móviles Android

    Args:
        args: Argumentos parseados
    """
    from ultralytics import YOLO

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_best_model()

    logger.info(f"Exportando modelo: {model_path}")

    model = YOLO(str(model_path))

    # Exportar al formato especificado
    export_path = model.export(
        format=args.export_format,
        imgsz=args.imgsz,
        half=args.half,              # FP16 para reducir tamaño
        dynamic=args.dynamic,        # Tamaños de entrada dinámicos
        simplify=True,               # Simplificar el grafo
    )

    logger.info(f"Modelo exportado a: {export_path}")


def main():
    """
    Punto de entrada principal del script.
    """
    parser = argparse.ArgumentParser(
        description='Inferencia con modelo YOLOv8/v11 entrenado para cajas VR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Imagen individual
  python scripts/inference.py --source imagen.jpg

  # Directorio de imágenes
  python scripts/inference.py --source carpeta/

  # Video
  python scripts/inference.py --source video.mp4

  # Webcam
  python scripts/inference.py --source 0

  # Con umbral de confianza específico
  python scripts/inference.py --source imagen.jpg --conf 0.5

  # Exportar modelo a ONNX
  python scripts/inference.py --export --export-format onnx
        """
    )

    # Argumentos principales
    parser.add_argument(
        '--source', '-s',
        type=str,
        default=None,
        help='Fuente: imagen, directorio, video, o número de webcam'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Ruta al modelo .pt (default: busca el mejor disponible)'
    )

    # Parámetros de detección
    parser.add_argument(
        '--conf',
        type=float,
        default=0.65,
        help='Umbral de confianza mínimo (default: 0.65)'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='Umbral de IoU para NMS (default: 0.45)'
    )

    parser.add_argument(
        '--max-det',
        type=int,
        default=100,
        help='Máximo número de detecciones por imagen (default: 100)'
    )

    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Tamaño de imagen para inferencia (default: 640)'
    )

    # Opciones de salida
    parser.add_argument(
        '--save',
        action='store_true',
        default=True,
        help='Guardar imágenes con detecciones (default: True)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='No guardar imágenes con detecciones'
    )

    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Guardar resultados en formato YOLO (.txt)'
    )

    parser.add_argument(
        '--save-conf',
        action='store_true',
        help='Incluir confianza en archivos de texto'
    )

    parser.add_argument(
        '--save-crop',
        action='store_true',
        help='Guardar recortes de las detecciones'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Mostrar resultados en ventana (requiere display)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Mostrar información detallada'
    )

    # Hardware
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device: 0 para GPU, cpu para CPU (default: 0)'
    )

    # Exportación
    parser.add_argument(
        '--export',
        action='store_true',
        help='Exportar modelo en lugar de hacer inferencia'
    )

    parser.add_argument(
        '--export-format',
        type=str,
        default='onnx',
        choices=['onnx', 'torchscript', 'openvino', 'engine', 'coreml', 'tflite'],
        help='Formato de exportación (default: onnx)'
    )

    parser.add_argument(
        '--half',
        action='store_true',
        help='Exportar con FP16 (reduce tamaño del modelo)'
    )

    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Exportar con tamaños de entrada dinámicos'
    )

    args = parser.parse_args()

    # Procesar argumentos
    if args.no_save:
        args.save = False

    # Decidir qué hacer
    if args.export:
        export_model(args)
    elif args.source:
        run_inference(args)
    else:
        parser.print_help()
        print("\n⚠️  Debes especificar --source o --export")
        sys.exit(1)


if __name__ == '__main__':
    main()
