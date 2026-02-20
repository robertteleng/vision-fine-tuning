# Learning — Fine-Tuning Studio

Guias paso a paso para aprender a usar el framework de fine-tuning.

## Contenido

| # | Guia | Descripcion |
|---|------|-------------|
| 01 | [Caso de Uso: Navegacion Asistida](01-caso-de-uso-navegacion-asistida.md) | Ejemplo real: detector de obstaculos para gafas Meta Aria |
| 02 | [Pipeline Completo](02-pipeline-completo.md) | De cero a produccion paso a paso |
| 03 | [Benchmark RTX 5060 Ti](03-benchmark-rtx5060ti.md) | Resultados reales: PyTorch vs ONNX vs TensorRT FP16 |
| 04 | [Auto-Anotacion con Grounding DINO](04-auto-anotacion-grounding-dino.md) | Como anotar automaticamente con descripcion textual |
| 05 | [Elegir Modelo YOLO26](05-elegir-modelo-yolo26.md) | Nano vs Small vs Medium vs Large: cual usar |
| 06 | [TensorRT Optimizacion](06-tensorrt-optimizacion.md) | Exportar y optimizar para maxima velocidad |

## Orden de Lectura Recomendado

1. **01** — Entender el caso de uso
2. **05** — Elegir el modelo correcto
3. **02** — Seguir el pipeline completo
4. **04** — Profundizar en auto-anotacion
5. **06** — Optimizar para produccion
6. **03** — Referencia de benchmark

## Resumen Rapido

```
Imagenes → Grounding DINO (auto-anotacion) → YOLO26m (fine-tune) → TensorRT FP16 (331 FPS)
```

Todo el pipeline toma ~3-4 horas de cero a modelo en produccion.
