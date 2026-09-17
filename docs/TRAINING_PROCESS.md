# Fine-Tuning YOLO26 — Estrategia y Pipeline

Como hacer fine-tuning de YOLO26 para añadir clases custom sin perder las de COCO.

## Concepto clave

**YOLO reemplaza la cabeza de deteccion completa al entrenar.**
No se pueden "añadir" clases a un modelo existente. Para tener N clases
en el modelo final, hay que entrenar con N clases en un solo dataset.

Esto implica:
- `freeze` congela features del backbone, pero NO preserva clases
- Entrenar solo con clases custom = perder las 80 clases de COCO
- Re-entrenar clases de COCO con datos diferentes = degradar rendimiento original

## Estrategia: dataset combinado

Crear un dataset que combine:
1. **Subset de COCO** con clases relevantes para tu caso de uso
2. **Dataset custom** con las clases nuevas que COCO no tiene

### Por que subset y no todo COCO

- COCO completo: 118K train + 5K val = ~20GB
- Subset: ~10K train + ~2K val = ~4GB
- El modelo base ya fue pre-entrenado en COCO completo — solo necesitamos
  suficientes ejemplos para que no "olvide" esas clases al añadir las nuevas
- Las clases de COCO que no son relevantes (ej: pizza, toaster) añadirian ruido

## Pipeline completo

```
1. Elegir las clases COCO relevantes para el caso de uso
2. Fijar las imagenes por ID en data_manifest/          (COCO 2017 + Open Images V7)
3. Construir el dataset combinado con IDs remapeados    → scripts/build_dataset.py
4. Entrenar YOLO26 con el dataset combinado             → scripts/train.py
5. Evaluar por clase y exportar a TensorRT              → scripts/benchmark.py
```

### Construir el dataset

```bash
uv sync --extra datasets
uv run python scripts/build_dataset.py --out data/nav_combined --download
```

- Las imagenes vienen de la cache de FiftyOne; las que falten se descargan por ID.
- Las etiquetas se convierten desde las anotaciones originales (JSON de COCO, CSV de Open Images).
  No se exporta desde FiftyOne, asi el resultado depende solo de los manifiestos.
- Se conservan las cajas `iscrowd` de COCO y las `IsGroupOf` de Open Images.
- Las clases de Open Images van a los IDs 18-23, detras de las 18 de COCO.
- `--verify <dataset>` compara las etiquetas con un dataset existente. Reconstruido desde cero, coincide
  con el usado para entrenar en las 14.179 imagenes.

**Por que manifiestos y no una regla.** Las imagenes COCO son las primeras por ID que contienen alguna
clase de navegacion, pero las de Open Images salieron de una descarga anterior que no se puede
reproducir con una regla. Fijar los IDs garantiza el mismo dataset.

### Entrenar

```bash
uv run python scripts/train.py --model yolo26n.pt --data data/nav_combined/dataset.yaml
```

Sin `freeze`: el modelo tiene que aprender la nueva cabeza de deteccion con todas las clases.
La receta completa esta en `config.yaml` (50 epocas, patience 20, close_mosaic 10, seed 42).

## Ejemplo: nav_combined (navegacion asistida)

### 24 clases finales

**18 de COCO (IDs 0-17):**
```
person, bicycle, car, motorcycle, bus, truck,
traffic light, fire hydrant, stop sign,
bench, chair, dog, cat,
backpack, umbrella, handbag, suitcase, potted plant
```

**6 custom de Open Images V7 (IDs 18-23):**
```
Door, Stairs, Street light, Traffic sign, Tree, Wheelchair
```

### Datos del dataset

| Split | COCO | Custom | Total |
|-------|------|--------|-------|
| train | 8,000 | 3,608 | 11,608 |
| val | 2,000 | 571 | 2,571 |

### Resultados

Estan en el [README](../README.md#results), generados desde `benchmarks/`. Resumen del entrenamiento
(`benchmarks/training/summary.json`):

| Modelo | Mejor epoca | mAP50 val | mAP50-95 val | Tiempo (RTX 5060 Ti) |
|--------|-------------|-----------|--------------|----------------------|
| YOLO26s | 37/50 | 0.470 | 0.311 | ~100 min |
| YOLO26n | 50/50 | 0.395 | 0.260 | ~61 min |

Las clases con pocas instancias de validacion dan mAP ruidoso: `Stairs` tiene 45 instancias en 36
imagenes y `Street light` 40 en 7.

## Mejoras pendientes

1. **Mas datos para las clases debiles.** Para `Stairs` ya esta medido: 1.000 imagenes mas suben su
   AP50 de 0.464 a 0.590, o a 0.543 si se autoanotan con Grounding DINO
   (ver [EXPERIMENT_STAIRS.md](EXPERIMENT_STAIRS.md)). `Street light` necesita bastantes mas imagenes de validacion.
2. **Mas epocas para nano:** su mejor epoca fue la ultima.
3. **Etiquetas parciales:** las imagenes de Open Images no etiquetan personas ni coches como clases COCO,
   y las de COCO no etiquetan puertas ni escaleras. Completarlas (por ejemplo con autoanotacion) quitaria
   falsos negativos al entrenar.

## Herramientas

- **FiftyOne**: Descarga de COCO y Open Images V7 con filtros por clase
- **Ultralytics YOLO26**: Entrenamiento, export, inferencia
- **TensorRT**: export FP16 (y experimentos INT8, ver `BENCHMARK_METHODOLOGY.md`)
