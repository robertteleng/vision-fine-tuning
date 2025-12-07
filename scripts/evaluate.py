#!/usr/bin/env python3
"""
Evaluación de métricas del modelo YOLO entrenado.

Calcula mAP, precision, recall y F1 sobre el dataset de validación.

Uso:
    python scripts/evaluate.py
    python scripts/evaluate.py --model runs/train/exp/weights/best.pt
    python scripts/evaluate.py --data data/pillar.yaml --conf 0.5
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


def evaluate(args):
    """Ejecuta la evaluación del modelo."""
    from ultralytics import YOLO

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

    logger.info(f"Cargando modelo: {model_path}")
    model = YOLO(str(model_path))

    # Buscar dataset config
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = PROJECT_ROOT / 'data' / 'pillar.yaml'

    if not data_path.exists():
        logger.error(f"Dataset config no encontrado: {data_path}")
        sys.exit(1)

    logger.info(f"Dataset: {data_path}")
    logger.info("=" * 60)
    logger.info("EJECUTANDO EVALUACIÓN")
    logger.info("=" * 60)

    # Ejecutar validación
    results = model.val(
        data=str(data_path),
        conf=args.conf,
        iou=args.iou,
        split=args.split,
        save_json=args.save_json,
        verbose=True,
    )

    # Mostrar resultados
    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS")
    logger.info("=" * 60)

    metrics = {
        'Precision': results.box.mp,
        'Recall': results.box.mr,
        'mAP@50': results.box.map50,
        'mAP@50-95': results.box.map,
    }

    for name, value in metrics.items():
        logger.info(f"{name:15}: {value:.4f} ({value*100:.1f}%)")

    # F1 score
    if results.box.mp > 0 and results.box.mr > 0:
        f1 = 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr)
        logger.info(f"{'F1 Score':15}: {f1:.4f} ({f1*100:.1f}%)")

    logger.info("=" * 60)

    # Métricas por clase
    if hasattr(results.box, 'ap_class_index') and len(results.box.ap_class_index) > 1:
        logger.info("\nMétricas por clase:")
        for i, cls_idx in enumerate(results.box.ap_class_index):
            cls_name = model.names[int(cls_idx)]
            logger.info(f"  {cls_name}: mAP50={results.box.ap50[i]:.3f}, mAP50-95={results.box.ap[i]:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluar modelo YOLO')

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Ruta al modelo .pt'
    )

    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='Ruta al archivo YAML del dataset'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.001,
        help='Umbral de confianza (default: 0.001 para evaluación)'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.6,
        help='Umbral de IoU para mAP (default: 0.6)'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Split a evaluar (default: val)'
    )

    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Guardar resultados en formato COCO JSON'
    )

    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
