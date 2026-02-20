# Benchmark: YOLO26m en RTX 5060 Ti 16GB

## Fecha: 20 Febrero 2026

## Hardware

| Componente | Especificacion |
|------------|----------------|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| VRAM | 16 GB GDDR7 |
| Driver | 590.48.01 |
| CUDA | 12.8 |
| TensorRT | 10.15.1.29 |
| PyTorch | 2.9.1+cu128 |
| Ultralytics | 8.4.7 |

## Modelo

| Propiedad | Valor |
|-----------|-------|
| Modelo | yolo26m.pt |
| Parametros | 22M |
| Tamano PyTorch | 42 MB |
| Tamano ONNX | 78 MB |
| Tamano TensorRT FP16 | 41 MB |
| Input | 640x640 RGB |

## Resultados de Inferencia

Benchmark con imagen dummy 640x640, 200 iteraciones, 20 warmup.

| Formato | Latencia Media | Std | FPS | vs PyTorch | Min | Max |
|---------|---------------|-----|-----|------------|-----|-----|
| PyTorch | 8.86 ms | 0.19 ms | 112.9 | 1.0x | 8.67 ms | 9.61 ms |
| ONNX (CPU)* | 177.06 ms | 10.55 ms | 5.6 | 0.1x | 171.31 ms | 318.64 ms |
| **TensorRT FP16** | **3.02 ms** | **0.03 ms** | **331.3** | **2.9x** | **2.97 ms** | **3.16 ms** |

*ONNX corrio en CPU porque faltaba onnxruntime-gpu. No es representativo del rendimiento ONNX en GPU.

## Performance de GPU Durante Benchmark

Datos capturados con `nvidia-smi dmon` a intervalos de 1 segundo.

### Estado Idle (antes del benchmark)

| Metrica | Valor |
|---------|-------|
| Power | 4 W |
| Temperatura | 31 C |
| GPU Utilization | 0% |
| Memory Utilization | 0% |
| VRAM usado | 364 MB |
| Clock GPU | 180 MHz |
| Clock Memoria | 405 MHz |

### PyTorch Inference (pico)

| Metrica | Valor |
|---------|-------|
| Power | 110 W |
| Temperatura | 45 C |
| GPU Utilization | 80% |
| Memory Utilization | 30% |
| VRAM usado | 825 MB |
| Clock GPU | 2820 MHz |
| Clock Memoria | 13801 MHz |

### TensorRT FP16 Inference (pico)

| Metrica | Valor |
|---------|-------|
| Power | 52 W |
| Temperatura | 36 C |
| GPU Utilization | 33% |
| Memory Utilization | 10% |
| VRAM usado | 949 MB |
| Clock GPU | 2812 MHz |
| Clock Memoria | 13801 MHz |

## Analisis

### TensorRT FP16 vs PyTorch

- **2.9x mas rapido** en latencia
- **53% menos consumo** de potencia (52W vs 110W)
- **58% menos utilizacion** GPU (33% vs 80%)
- **Consistencia extrema**: std de solo 0.03ms vs 0.19ms
- **Rango min-max**: 2.97-3.16ms (variacion de 0.19ms)

### Por que TensorRT es tan rapido

1. **Fusion de capas**: combina Conv+BN+ReLU en una sola operacion
2. **FP16 precision**: la mitad de bits = el doble de throughput
3. **Kernel auto-tuning**: elige los mejores kernels CUDA para tu GPU especifica
4. **Memory layout**: optimiza el layout de tensores para el hardware
5. **Graph optimization**: elimina operaciones redundantes

### VRAM disponible para batch processing

Con solo 949 MB usados de 16 GB, queda margen para:

- Batch de 8-16 imagenes simultaneas
- Pipeline de pre/post-procesado en GPU
- Multiples modelos simultaneos (deteccion + segmentacion)

## Comando para Reproducir

```bash
# Exportar modelos
python scripts/export_tensorrt.py --format onnx
python scripts/export_tensorrt.py --format engine --half

# Correr benchmark
python scripts/benchmark.py --model models/yolo26m.pt --iterations 200 --warmup 20

# Monitor de GPU en paralelo
nvidia-smi dmon -s pucvmet -d 1 -c 120 > gpu_monitor.log &
```
