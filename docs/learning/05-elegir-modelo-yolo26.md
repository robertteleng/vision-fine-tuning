# Como Elegir el Modelo YOLO26 Correcto

## Los Modelos YOLO26

YOLO26 viene en 5 tamanos. Cada uno es un modelo completo, NO una version recortada de otro.

| Modelo | Parametros | Tamano .pt | FPS (RTX 5060 Ti)* | mAP COCO |
|--------|-----------|------------|---------------------|----------|
| yolo26n.pt | 2.7M | ~6 MB | ~800+ | ~37 |
| yolo26s.pt | 9.6M | ~19 MB | ~500+ | ~44 |
| **yolo26m.pt** | **22M** | **42 MB** | **331 (TensorRT)** | **~50** |
| yolo26l.pt | 43M | ~86 MB | ~200 | ~53 |
| yolo26x.pt | 78M | ~156 MB | ~120 | ~55 |

*FPS estimados con TensorRT FP16 en RTX 5060 Ti 16GB. Solo yolo26m confirmado con benchmark real.

## Pregunta Clave: Small o Medium?

**Se entrena directamente en el tamano que quieres usar en produccion.**

No se puede:
- Entrenar en medium y "comprimir" a small
- Entrenar en large y "destilar" a small (esto se llama Knowledge Distillation y es un proceso completamente diferente)

**Cada modelo es independiente:**

```bash
# Entrenar modelo small
python scripts/train.py --data dataset.yaml --model yolo26s.pt

# Entrenar modelo medium
python scripts/train.py --data dataset.yaml --model yolo26m.pt
```

Ambos parten de pesos pre-entrenados en COCO, pero tienen arquitecturas distintas (diferente numero de capas, canales, etc.).

## Cuando Usar Cada Tamano

### yolo26n (Nano) — Para edge extremo
- **Caso**: Movil, Raspberry Pi, Arduino con NPU
- **FPS**: Muy alto, incluso en CPU
- **Precision**: La mas baja, pero aceptable para objetos grandes
- **Ejemplo**: Detector simple de 1-2 clases en movil

### yolo26s (Small) — Para dispositivos limitados
- **Caso**: Jetson Nano, movil gama media, gafas AR
- **FPS**: Alto en GPU movil
- **Precision**: Buena para la mayoria de aplicaciones
- **Ejemplo**: Navegacion asistida en gafas con GPU limitada

### yolo26m (Medium) — Sweet spot
- **Caso**: GPU dedicada (RTX, Jetson Orin, servidor)
- **FPS**: 331 FPS con TensorRT en RTX 5060 Ti
- **Precision**: Excelente — suficiente para la mayoria de tareas
- **Ejemplo**: Nuestro caso con RTX 5060 Ti

### yolo26l (Large) — Para maxima precision
- **Caso**: Servidor con GPU potente
- **FPS**: ~200 FPS con TensorRT
- **Precision**: Muy alta, especialmente objetos pequenos
- **Ejemplo**: Inspeccion industrial, seguridad critica

### yolo26x (Extra Large) — Para investigacion
- **Caso**: GPU de datacenter (A100, H100)
- **FPS**: ~120 FPS con TensorRT
- **Precision**: La mas alta posible
- **Ejemplo**: Benchmark academico, datasets muy complejos

## Para RTX 5060 Ti 16GB

**Recomendacion: yolo26m.pt**

Razon:
1. 331 FPS con TensorRT — mas que suficiente para tiempo real
2. Solo usa 949 MB de 16 GB VRAM — queda margen enorme
3. 22M parametros — buen equilibrio precision/velocidad
4. El batch de entrenamiento puede ser grande (-1 auto = probablemente 32-64)

Si necesitas aun mas velocidad (ej: procesar 8 camaras simultaneas):
- Usar yolo26s.pt — menos preciso pero ~500 FPS

## Estrategia Recomendada: Entrenar Ambos y Comparar

```bash
# Entrenar medium
python scripts/train.py \
    --data data/dataset_nav/dataset.yaml \
    --model yolo26m.pt \
    --name nav_medium_v1

# Entrenar small
python scripts/train.py \
    --data data/dataset_nav/dataset.yaml \
    --model yolo26s.pt \
    --name nav_small_v1

# Evaluar ambos
python scripts/evaluate.py --model runs/train/nav_medium_v1/weights/best.pt
python scripts/evaluate.py --model runs/train/nav_small_v1/weights/best.pt

# Benchmark ambos
python scripts/benchmark.py --model runs/train/nav_medium_v1/weights/best.pt
python scripts/benchmark.py --model runs/train/nav_small_v1/weights/best.pt
```

Despues decides basandote en datos reales, no en intuicion.

## Variantes por Tarea

Ademas del tamano, YOLO26 tiene variantes por tarea:

| Sufijo | Tarea | Ejemplo |
|--------|-------|---------|
| (nada) | Deteccion (bounding boxes) | `yolo26m.pt` |
| `-seg` | Segmentacion (mascaras pixel) | `yolo26m-seg.pt` |
| `-cls` | Clasificacion (imagen completa) | `yolo26m-cls.pt` |
| `-pose` | Estimacion de pose (keypoints) | `yolo26m-pose.pt` |
| `-obb` | Oriented Bounding Boxes | `yolo26m-obb.pt` |

Para navegacion asistida, **deteccion** (`yolo26m.pt`) es suficiente.
Si quisieras saber la forma exacta del obstaculo, usarias **segmentacion** (`yolo26m-seg.pt`).

## Resumen Rapido

```
¿Hardware limitado (movil, Jetson)?  → yolo26s.pt
¿GPU dedicada (RTX, servidor)?      → yolo26m.pt
¿Necesitas maxima precision?         → yolo26l.pt
¿Solo investigacion/benchmark?       → yolo26x.pt
```
