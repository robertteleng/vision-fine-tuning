#!/usr/bin/env python3
"""
Exportación de modelo YOLO a TensorRT para inferencia optimizada.

Formatos disponibles:
- engine: TensorRT (NVIDIA GPUs, más rápido)
- onnx: ONNX (portable)
- torchscript: TorchScript

Uso:
    python scripts/export_tensorrt.py
    python scripts/export_tensorrt.py --format engine --half
    python scripts/export_tensorrt.py --model runs/train/exp/weights/best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def find_best_model() -> Path:
    """Busca el mejor modelo disponible."""
    models_dir = PROJECT_ROOT / 'models'
    if models_dir.exists():
        versioned_models = sorted(models_dir.glob('vr_boxes_*.pt'), reverse=True)
        if versioned_models:
            return versioned_models[0]

    runs_dir = PROJECT_ROOT / 'runs' / 'train'
    if runs_dir.exists():
        experiments = sorted(runs_dir.glob('*/weights/best.pt'), reverse=True)
        if experiments:
            return experiments[0]

    raise FileNotFoundError("No se encontró ningún modelo entrenado.")


def export_model(args):
    """Exporta el modelo al formato especificado."""
    from ultralytics import YOLO
    import torch

    # Verificar CUDA para TensorRT
    if args.format == 'engine' and not torch.cuda.is_available():
        logger.error("TensorRT requiere GPU NVIDIA con CUDA")
        sys.exit(1)

    # Cargar modelo
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

    logger.info(f"Modelo a exportar: {model_path}")
    model = YOLO(str(model_path))

    # Info del formato
    format_info = {
        'engine': 'TensorRT (optimizado para NVIDIA GPUs)',
        'onnx': 'ONNX (formato portable)',
        'torchscript': 'TorchScript (PyTorch nativo)',
        'openvino': 'OpenVINO (Intel)',
        'coreml': 'CoreML (Apple)',
        'tflite': 'TensorFlow Lite (móviles)',
    }

    logger.info("=" * 60)
    logger.info(f"EXPORTANDO A: {format_info.get(args.format, args.format)}")
    logger.info("=" * 60)

    export_args = {
        'format': args.format,
        'imgsz': args.imgsz,
        'half': args.half,
        'dynamic': args.dynamic,
        'simplify': True,
        'verbose': True,
    }

    # Opciones específicas de TensorRT
    if args.format == 'engine':
        export_args['device'] = 0
        if args.workspace:
            export_args['workspace'] = args.workspace
        logger.info(f"  Precisión: {'FP16' if args.half else 'FP32'}")
        logger.info(f"  Tamaño imagen: {args.imgsz}")

    # Exportar
    try:
        export_path = model.export(**export_args)
        logger.info("\n" + "=" * 60)
        logger.info("EXPORTACIÓN COMPLETADA")
        logger.info("=" * 60)
        logger.info(f"Modelo exportado: {export_path}")

        # Mostrar tamaño
        export_file = Path(export_path)
        if export_file.exists():
            size_mb = export_file.stat().st_size / (1024 * 1024)
            logger.info(f"Tamaño: {size_mb:.1f} MB")

        # Instrucciones de uso
        logger.info("\nPara usar el modelo exportado:")
        if args.format == 'engine':
            logger.info(f"  python scripts/inference.py --source video.mp4 --model {export_path}")
        elif args.format == 'onnx':
            logger.info(f"  from ultralytics import YOLO")
            logger.info(f"  model = YOLO('{export_path}')")
            logger.info(f"  results = model('imagen.jpg')")

        return export_path

    except Exception as e:
        logger.error(f"Error en exportación: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Exportar modelo YOLO a TensorRT u otros formatos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Exportar a TensorRT FP16 (recomendado para producción)
  python scripts/export_tensorrt.py --format engine --half

  # Exportar a ONNX
  python scripts/export_tensorrt.py --format onnx

  # Exportar modelo específico
  python scripts/export_tensorrt.py --model models/best.pt --format engine
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Ruta al modelo .pt'
    )

    parser.add_argument(
        '--format', '-f',
        type=str,
        default='engine',
        choices=['engine', 'onnx', 'torchscript', 'openvino', 'coreml', 'tflite'],
        help='Formato de exportación (default: engine/TensorRT)'
    )

    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Tamaño de imagen (default: 640)'
    )

    parser.add_argument(
        '--half',
        action='store_true',
        help='Exportar en FP16 (reduce tamaño, más rápido)'
    )

    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Permitir tamaños de entrada dinámicos'
    )

    parser.add_argument(
        '--workspace',
        type=int,
        default=4,
        help='Workspace de TensorRT en GB (default: 4)'
    )

    args = parser.parse_args()
    export_model(args)


if __name__ == '__main__':
    main()
