#!/usr/bin/env python3
"""
Benchmark de velocidad de inferencia para diferentes formatos de modelo.

Compara: PyTorch (.pt), ONNX (.onnx), TensorRT (.engine)

Uso:
    python scripts/benchmark.py
    python scripts/benchmark.py --model runs/train/exp/weights/best.pt
    python scripts/benchmark.py --iterations 200
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

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


def benchmark_model(model_path: Path, imgsz: int, iterations: int, warmup: int):
    """Ejecuta benchmark en un modelo."""
    from ultralytics import YOLO
    import torch

    logger.info(f"Cargando: {model_path.name}")
    model = YOLO(str(model_path))

    # Crear imagen dummy
    dummy_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    logger.info(f"  Warmup ({warmup} iteraciones)...")
    for _ in range(warmup):
        model(dummy_img, verbose=False)

    # Sincronizar GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    logger.info(f"  Benchmark ({iterations} iteraciones)...")
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        model(dummy_img, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    # Estadísticas
    times_ms = np.array(times) * 1000
    return {
        'mean': np.mean(times_ms),
        'std': np.std(times_ms),
        'min': np.min(times_ms),
        'max': np.max(times_ms),
        'median': np.median(times_ms),
        'fps': 1000 / np.mean(times_ms),
    }


def run_benchmark(args):
    """Ejecuta benchmark comparativo."""
    import torch

    # Info del sistema
    logger.info("=" * 60)
    logger.info("BENCHMARK DE INFERENCIA")
    logger.info("=" * 60)

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"VRAM: {gpu_mem:.1f} GB")
    else:
        logger.info("GPU: No disponible (usando CPU)")

    logger.info(f"Tamaño imagen: {args.imgsz}x{args.imgsz}")
    logger.info(f"Iteraciones: {args.iterations}")
    logger.info("-" * 60)

    # Buscar modelo base
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

    # Modelos a comparar
    models_to_test = []

    # PyTorch
    if model_path.suffix == '.pt':
        models_to_test.append(('PyTorch', model_path))

    # Buscar versiones exportadas
    parent = model_path.parent
    stem = model_path.stem

    onnx_path = parent / f"{stem}.onnx"
    if onnx_path.exists():
        models_to_test.append(('ONNX', onnx_path))

    engine_path = parent / f"{stem}.engine"
    if engine_path.exists():
        models_to_test.append(('TensorRT', engine_path))

    # Si solo hay .pt, añadirlo
    if not models_to_test:
        models_to_test.append(('PyTorch', model_path))

    # Ejecutar benchmarks
    results = {}
    for name, path in models_to_test:
        try:
            logger.info(f"\n{name}:")
            stats = benchmark_model(path, args.imgsz, args.iterations, args.warmup)
            results[name] = stats
            logger.info(f"  Media: {stats['mean']:.2f}ms (±{stats['std']:.2f}ms)")
            logger.info(f"  FPS: {stats['fps']:.1f}")
        except Exception as e:
            logger.warning(f"  Error: {e}")

    # Tabla comparativa
    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS COMPARATIVOS")
    logger.info("=" * 60)

    logger.info(f"{'Formato':<12} {'Media (ms)':<12} {'Std (ms)':<10} {'FPS':<8} {'Min (ms)':<10} {'Max (ms)'}")
    logger.info("-" * 60)

    baseline_fps = None
    for name, stats in results.items():
        if baseline_fps is None:
            baseline_fps = stats['fps']
            speedup = ""
        else:
            speedup = f" ({stats['fps']/baseline_fps:.1f}x)"

        logger.info(
            f"{name:<12} {stats['mean']:>8.2f}    {stats['std']:>8.2f}  "
            f"{stats['fps']:>6.1f}{speedup:<6} {stats['min']:>8.2f}    {stats['max']:.2f}"
        )

    logger.info("=" * 60)

    # Recomendaciones
    if len(results) == 1 and 'PyTorch' in results:
        logger.info("\nPara comparar formatos, primero exporta el modelo:")
        logger.info("  python scripts/export_tensorrt.py --format engine --half")
        logger.info("  python scripts/export_tensorrt.py --format onnx")

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark de velocidad de inferencia')

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Ruta al modelo .pt (buscará versiones exportadas automáticamente)'
    )

    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Tamaño de imagen (default: 640)'
    )

    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=100,
        help='Número de iteraciones (default: 100)'
    )

    parser.add_argument(
        '--warmup', '-w',
        type=int,
        default=10,
        help='Iteraciones de warmup (default: 10)'
    )

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == '__main__':
    main()
