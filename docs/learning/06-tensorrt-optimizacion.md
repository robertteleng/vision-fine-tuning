# TensorRT: Exportacion y Optimizacion

## Que es TensorRT

TensorRT es el compilador de modelos de NVIDIA. Toma un modelo PyTorch/ONNX y lo optimiza especificamente para tu GPU, produciendo un motor de inferencia mucho mas rapido.

## Por Que Usar TensorRT

| Metrica | PyTorch | TensorRT FP16 | Mejora |
|---------|---------|----------------|--------|
| Latencia | 8.86 ms | 3.02 ms | 2.9x |
| FPS | 112.9 | 331.3 | 2.9x |
| Power | 110 W | 52 W | 53% menos |
| GPU Util | 80% | 33% | 58% menos |
| Consistencia (std) | 0.19 ms | 0.03 ms | 6x mas estable |

## Flujo de Exportacion

```
yolo26m.pt (PyTorch, 42MB)
  ↓ export --format onnx
yolo26m.onnx (ONNX, 78MB)
  ↓ export --format engine --half
yolo26m.engine (TensorRT FP16, 41MB)
```

## Comandos

### Exportar a TensorRT FP16

```bash
python scripts/export_tensorrt.py --format engine --half
```

Opciones:
- `--half` — FP16 precision (recomendado para inferencia)
- `--imgsz 640` — tamano de entrada (debe coincidir con entrenamiento)
- `--model models/best.pt` — modelo especifico
- `--workspace 4` — GB de workspace para TensorRT

### Exportar a ONNX

```bash
python scripts/export_tensorrt.py --format onnx
```

ONNX es portable — funciona en cualquier plataforma con ONNX Runtime.

### Exportar a otros formatos

```bash
# TorchScript (portable PyTorch)
python scripts/export_tensorrt.py --format torchscript

# OpenVINO (Intel CPUs/GPUs)
python scripts/export_tensorrt.py --format openvino

# CoreML (Apple Silicon)
python scripts/export_tensorrt.py --format coreml

# TFLite (movil Android/iOS)
python scripts/export_tensorrt.py --format tflite
```

## FP16 vs FP32

| Precision | Bits | Tamano | Velocidad | Precision (mAP) |
|-----------|------|--------|-----------|------------------|
| FP32 | 32 | Grande | Lento | 100% (baseline) |
| **FP16** | 16 | **50%** | **~2x** | **~99.5%** |
| INT8 | 8 | 25% | ~4x | ~98% (necesita calibracion) |

**FP16 es el sweet spot**: practicamente la misma precision con el doble de velocidad.

## El Archivo .engine

El archivo `.engine` es especifico para:
- Tu GPU exacta (RTX 5060 Ti)
- Tu version de TensorRT
- Tu tamano de entrada (640x640)

**NO es portable** — si cambias de GPU, tienes que re-exportar.

## Que Optimizaciones Hace TensorRT

1. **Layer Fusion**: Conv + BatchNorm + Activation → una sola operacion
2. **Kernel Auto-Tuning**: prueba todos los kernels CUDA y elige el mas rapido
3. **Precision Calibration**: convierte FP32 → FP16 sin perder precision
4. **Memory Optimization**: reusa memoria entre capas que no se solapan
5. **Graph Optimization**: elimina operaciones redundantes

## Verificar Que Funciona

```bash
# Benchmark comparativo
python scripts/benchmark.py --model models/yolo26m.pt --iterations 200

# Inferencia con TensorRT
python scripts/inference.py --source imagen.jpg --model models/yolo26m.engine

# Monitor GPU durante inferencia
nvidia-smi dmon -s pucvmet -d 1
```

## Troubleshooting

| Problema | Solucion |
|----------|----------|
| CUDA not available | Instalar drivers NVIDIA + CUDA toolkit |
| TensorRT not found | `pip install tensorrt` |
| Export timeout | Aumentar `--workspace` a 8 |
| OOM durante export | Cerrar otras apps que usen GPU |
| Engine no funciona en otra GPU | Re-exportar en la GPU destino |
| Precision diferente a PyTorch | Normal con FP16, diferencia <0.5% |
