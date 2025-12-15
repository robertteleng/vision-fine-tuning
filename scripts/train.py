#!/usr/bin/env python3
"""
=============================================================================
Script de Entrenamiento - Fine-Tuning YOLOv8/v11 para Cajas VR
=============================================================================

Este script realiza el fine-tuning de un modelo YOLO pre-entrenado para
detectar cajas específicas en capturas de VR.

Uso:
    python scripts/train.py                    # Usa config.yaml por defecto
    python scripts/train.py --config mi_config.yaml
    python scripts/train.py --epochs 50        # Override de parámetros

Autor: [Tu nombre]
Fecha: Diciembre 2024
=============================================================================
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURACIÓN INICIAL
# =============================================================================

# Añadir el directorio raíz del proyecto al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configurar logging con formato detallado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_gpu_availability():
    """
    Verifica la disponibilidad de GPU y muestra información del hardware.

    Esta función es crucial para debugging cuando el entrenamiento es lento
    o cuando hay problemas de memoria.

    Returns:
        bool: True si hay GPU CUDA disponible, False en caso contrario
    """
    import torch

    logger.info("=" * 60)
    logger.info("VERIFICACIÓN DE HARDWARE")
    logger.info("=" * 60)

    # Información de PyTorch
    logger.info(f"PyTorch version: {torch.__version__}")

    # Verificar CUDA
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA disponible: {cuda_available}")

    if cuda_available:
        # Información de CUDA
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")

        # Información de cada GPU
        gpu_count = torch.cuda.device_count()
        logger.info(f"GPUs detectadas: {gpu_count}")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            logger.info(f"  GPU {i}: {props.name}")
            logger.info(f"         VRAM: {vram_gb:.1f} GB")
            logger.info(f"         Compute Capability: {props.major}.{props.minor}")

            # Advertencias específicas por GPU
            # ----------------------------------------------------------------
            # [GPU DEPENDIENTE] Ajusta estos valores según tu hardware
            # ----------------------------------------------------------------
            if vram_gb < 8:
                logger.warning(f"         ⚠️  VRAM limitada. Usa batch_size <= 8")
            elif vram_gb >= 16:
                logger.info(f"         ✅ VRAM amplia. Puedes usar batch_size 32-64")
    else:
        logger.warning("⚠️  NO HAY GPU CUDA DISPONIBLE")
        logger.warning("   El entrenamiento será MUY LENTO en CPU")
        logger.warning("   Verifica la instalación de CUDA y drivers NVIDIA")

    logger.info("=" * 60)
    return cuda_available


def load_config(config_path: str) -> dict:
    """
    Carga la configuración desde un archivo YAML.

    Args:
        config_path: Ruta al archivo de configuración YAML

    Returns:
        dict: Configuración cargada

    Raises:
        FileNotFoundError: Si el archivo no existe
        yaml.YAMLError: Si el archivo tiene formato inválido
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    logger.info(f"Cargando configuración desde: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def create_data_yaml(config: dict, output_path: Path) -> str:
    """
    Crea el archivo data.yaml necesario para YOLO.

    YOLO requiere un archivo YAML específico que define las rutas del dataset
    y las clases. Esta función lo genera dinámicamente basándose en config.yaml.

    Args:
        config: Diccionario de configuración
        output_path: Directorio donde guardar el archivo

    Returns:
        str: Ruta al archivo data.yaml creado
    """
    data_dir = PROJECT_ROOT / config.get('data_dir', 'data')

    # Estructura del data.yaml para YOLO
    # -----------------------------------------------------------------
    # NOTA: Las rutas deben ser absolutas para evitar problemas
    # -----------------------------------------------------------------
    data_yaml = {
        'path': str(data_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': config.get('nc', 2),
        'names': config.get('names', {0: 'vr_box_type_a', 1: 'vr_box_type_b'})
    }

    # Guardar el archivo
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Archivo data.yaml creado en: {yaml_path}")

    # Verificar que las carpetas existen
    for split in ['train', 'valid']:
        img_path = data_dir / split / 'images'
        label_path = data_dir / split / 'labels'

        if not img_path.exists():
            logger.warning(f"⚠️  Carpeta no encontrada: {img_path}")
            logger.warning(f"   Asegúrate de colocar tu dataset de Roboflow aquí")
        else:
            num_images = len(list(img_path.glob('*.[jJpP][pPnN][gG]')))
            num_labels = len(list(label_path.glob('*.txt')))
            logger.info(f"   {split}: {num_images} imágenes, {num_labels} labels")

    return str(yaml_path)


def get_experiment_name(base_name: str = None) -> str:
    """
    Genera un nombre único para el experimento con timestamp.

    Esto es importante para versionado de modelos y tracking de experimentos.

    Args:
        base_name: Nombre base opcional

    Returns:
        str: Nombre del experimento con timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if base_name:
        return f"{base_name}_{timestamp}"
    else:
        return f"vr_boxes_{timestamp}"


def train(args):
    """
    Función principal de entrenamiento.

    Esta función:
    1. Carga la configuración
    2. Prepara el dataset
    3. Inicializa el modelo
    4. Ejecuta el entrenamiento
    5. Guarda y reporta resultados

    Args:
        args: Argumentos de línea de comandos parseados
    """
    # =========================================================================
    # PASO 1: VERIFICAR HARDWARE
    # =========================================================================
    has_gpu = check_gpu_availability()

    if not has_gpu and not args.force_cpu:
        logger.error("No hay GPU disponible. Usa --force-cpu para entrenar en CPU (no recomendado)")
        sys.exit(1)

    # =========================================================================
    # PASO 2: CARGAR CONFIGURACIÓN
    # =========================================================================
    config = load_config(args.config)

    # Override de parámetros desde línea de comandos
    # -------------------------------------------------------------------------
    # Permite ajustar rápidamente sin editar el YAML
    # -------------------------------------------------------------------------
    if args.epochs:
        config['epochs'] = args.epochs
        logger.info(f"Override: epochs = {args.epochs}")

    if args.batch:
        config['batch'] = args.batch
        logger.info(f"Override: batch = {args.batch}")

    if args.imgsz:
        config['imgsz'] = args.imgsz
        logger.info(f"Override: imgsz = {args.imgsz}")

    # =========================================================================
    # PASO 3: PREPARAR DATASET
    # =========================================================================
    logger.info("Preparando dataset...")
    data_yaml_path = str(PROJECT_ROOT / 'data' / 'pillar.yaml')

    # =========================================================================
    # PASO 4: INICIALIZAR MODELO
    # =========================================================================
    # Importamos aquí para que los mensajes de verificación salgan primero
    from ultralytics import YOLO

    model_name = config.get('model', 'yolo12s.pt')
    logger.info(f"Cargando modelo base: {model_name}")

    # El modelo se descarga automáticamente si no existe localmente
    model = YOLO(model_name)

    # Información del modelo
    logger.info(f"Modelo cargado: {model.model_name if hasattr(model, 'model_name') else model_name}")

    # =========================================================================
    # PASO 5: CONFIGURAR NOMBRE DEL EXPERIMENTO
    # =========================================================================
    exp_name = args.name or config.get('name') or get_experiment_name()
    logger.info(f"Nombre del experimento: {exp_name}")

    # =========================================================================
    # PASO 6: EJECUTAR ENTRENAMIENTO
    # =========================================================================
    logger.info("=" * 60)
    logger.info("INICIANDO ENTRENAMIENTO")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Parámetros de entrenamiento
    # -------------------------------------------------------------------------
    # Todos los parámetros están documentados en config.yaml
    # Los más críticos para GPU:
    #   - batch: Reduce si hay OOM (Out of Memory)
    #   - imgsz: Reduce a 320 si es necesario
    #   - workers: Reduce si hay problemas de CPU
    # -------------------------------------------------------------------------

    training_params = {
        # Dataset
        'data': data_yaml_path,

        # Duración del entrenamiento
        'epochs': config.get('epochs', 100),
        'patience': config.get('patience', 20),

        # [GPU DEPENDIENTE] Parámetros críticos de memoria
        'batch': config.get('batch', 8),           # RTX 2060: 8, RTX 5060 Ti: 32
        'imgsz': config.get('imgsz', 640),         # 640 estándar, 1280 para más detalle
        'workers': config.get('workers', 4),       # RTX 2060: 4, RTX 5060 Ti: 8
        'cache': config.get('cache', False),       # False/ram

        # Learning rate
        'lr0': config.get('lr0', 0.01),
        'lrf': config.get('lrf', 0.01),
        'momentum': config.get('momentum', 0.937),
        'weight_decay': config.get('weight_decay', 0.0005),

        # Warmup
        'warmup_epochs': config.get('warmup_epochs', 3.0),
        'warmup_momentum': config.get('warmup_momentum', 0.8),
        'warmup_bias_lr': config.get('warmup_bias_lr', 0.1),

        # Augmentación
        'fliplr': config.get('fliplr', 0.5),
        'flipud': config.get('flipud', 0.0),
        'degrees': config.get('degrees', 10.0),
        'translate': config.get('translate', 0.1),
        'scale': config.get('scale', 0.5),
        'shear': config.get('shear', 2.0),
        'perspective': config.get('perspective', 0.0),
        'hsv_h': config.get('hsv_h', 0.015),
        'hsv_s': config.get('hsv_s', 0.7),
        'hsv_v': config.get('hsv_v', 0.4),
        'mosaic': config.get('mosaic', 1.0),
        'mixup': config.get('mixup', 0.0),
        'copy_paste': config.get('copy_paste', 0.0),

        # Guardado
        'project': str(PROJECT_ROOT / config.get('project', 'runs/train')),
        'name': exp_name,
        'save_period': config.get('save_period', 10),
        'save': config.get('save', True),
        'plots': config.get('plots', True),

        # Hardware
        'device': config.get('device', '0'),
        'amp': config.get('amp', True),            # Mixed precision (ahorra memoria)
        'deterministic': config.get('deterministic', False),
        'seed': config.get('seed', 42),

        # Optimizador
        'optimizer': config.get('optimizer', 'auto'),

        # Validación
        'val': config.get('val', True),

        # Verbose
        'verbose': config.get('verbose', True),
    }

    # Log de parámetros importantes
    logger.info("Parámetros de entrenamiento:")
    logger.info(f"  - Epochs: {training_params['epochs']}")
    logger.info(f"  - Batch size: {training_params['batch']}")
    logger.info(f"  - Image size: {training_params['imgsz']}")
    logger.info(f"  - Learning rate: {training_params['lr0']}")
    logger.info(f"  - Device: {training_params['device']}")
    logger.info(f"  - AMP (Mixed Precision): {training_params['amp']}")

    # -------------------------------------------------------------------------
    # ENTRENAMIENTO
    # -------------------------------------------------------------------------
    try:
        results = model.train(**training_params)

        logger.info("=" * 60)
        logger.info("ENTRENAMIENTO COMPLETADO")
        logger.info("=" * 60)

        # Mostrar métricas finales
        if results:
            logger.info("Métricas finales:")
            # Las métricas están en results.results_dict
            if hasattr(results, 'results_dict'):
                for key, value in results.results_dict.items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.4f}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("=" * 60)
            logger.error("ERROR: CUDA OUT OF MEMORY")
            logger.error("=" * 60)
            logger.error("Soluciones:")
            logger.error("  1. Reduce batch size en config.yaml (ej: batch: 4)")
            logger.error("  2. Reduce imgsz en config.yaml (ej: imgsz: 320)")
            logger.error("  3. Cierra otras aplicaciones que usen la GPU")
            logger.error("=" * 60)
        raise

    # =========================================================================
    # PASO 7: GUARDAR MODELO CON TIMESTAMP
    # =========================================================================
    # Además de best.pt y last.pt, guardamos una copia con timestamp
    models_dir = PROJECT_ROOT / 'models'
    models_dir.mkdir(exist_ok=True)

    best_model_path = PROJECT_ROOT / config.get('project', 'runs/train') / exp_name / 'weights' / 'best.pt'

    if best_model_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d")
        # Extraer nombre del modelo base (ej: "yolo12s" de "yolo12s.pt")
        base_model = Path(model_name).stem  # yolo12s.pt → yolo12s
        # Formato: {modelo_base}_pillars_{fecha}.pt  o  {modelo_base}+pillars_{fecha}.pt
        if args.coco:
            versioned_name = f"{base_model}+pillars_{timestamp}.pt"  # COCO + pillars
        else:
            versioned_name = f"{base_model}_pillars_{timestamp}.pt"  # solo pillars
        versioned_path = models_dir / versioned_name

        import shutil
        shutil.copy(best_model_path, versioned_path)

        logger.info(f"Modelo guardado con versionado: {versioned_path}")
        logger.info(f"Mejor modelo también en: {best_model_path}")

    # =========================================================================
    # PASO 8: RESUMEN FINAL
    # =========================================================================
    logger.info("=" * 60)
    logger.info("RESUMEN")
    logger.info("=" * 60)
    logger.info(f"Directorio de resultados: {PROJECT_ROOT / config.get('project', 'runs/train') / exp_name}")
    logger.info(f"Modelo versionado: {models_dir}")
    logger.info("")
    logger.info("Próximos pasos:")
    logger.info("  1. Revisa las métricas en TensorBoard:")
    logger.info(f"     tensorboard --logdir {PROJECT_ROOT / config.get('project', 'runs/train')}")
    logger.info("  2. Prueba el modelo con inference.py:")
    logger.info(f"     python scripts/inference.py --source path/to/image.jpg")
    logger.info("  3. Documenta los resultados en docs/benchmarks.md")
    logger.info("=" * 60)


def main():
    """
    Punto de entrada principal del script.

    Parsea argumentos de línea de comandos y ejecuta el entrenamiento.
    """
    parser = argparse.ArgumentParser(
        description='Fine-tuning de YOLOv8/v11 para detección de cajas VR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python scripts/train.py                      # Configuración por defecto
  python scripts/train.py --epochs 50          # Override de épocas
  python scripts/train.py --batch 4            # Batch más pequeño para GPUs limitadas
  python scripts/train.py --config mi_config.yaml  # Usar otra configuración
        """
    )

    # Argumentos de configuración
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=str(PROJECT_ROOT / 'config.yaml'),
        help='Ruta al archivo de configuración YAML (default: config.yaml)'
    )

    # Overrides de parámetros comunes
    # -------------------------------------------------------------------------
    # Estos permiten ajustar rápidamente sin editar el archivo de configuración
    # -------------------------------------------------------------------------
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='Número de épocas (override de config.yaml)'
    )

    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=None,
        help='Batch size (override de config.yaml). RTX 2060: 8, RTX 5060 Ti: 32'
    )

    parser.add_argument(
        '--imgsz',
        type=int,
        default=None,
        help='Tamaño de imagen (override de config.yaml)'
    )

    parser.add_argument(
        '--name', '-n',
        type=str,
        default=None,
        help='Nombre del experimento (default: auto-generado con timestamp)'
    )

    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Forzar entrenamiento en CPU (MUY lento, solo para testing)'
    )

    parser.add_argument(
        '--coco',
        action='store_true',
        help='Indica que el modelo incluye clases COCO + pillars (cambia nomenclatura)'
    )

    args = parser.parse_args()

    # Ejecutar entrenamiento
    try:
        train(args)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Entrenamiento interrumpido por el usuario")
        logger.info("Los checkpoints guardados están en runs/train/")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        raise


if __name__ == '__main__':
    main()
