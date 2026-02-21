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
1. Identificar clases COCO relevantes para tu caso de uso
2. Descargar COCO subset via FiftyOne            → data/coco_nav/
3. Preparar dataset custom (Open Images, manual)  → data/nav_custom/
4. Combinar ambos con IDs remapeados              → data/nav_combined/
5. Entrenar YOLO26 con dataset combinado
6. Exportar a TensorRT FP16
7. Testear con video real
```

### Paso 1 — Descargar COCO subset

```python
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=["person", "car", "bus", ...],  # Solo clases relevantes
    max_samples=8000,
)
```

### Paso 2 — Exportar a formato YOLO

```python
from fiftyone import ViewField as F

view = dataset.filter_labels(
    "ground_truth",
    F("label").is_in(MY_CLASSES),
    only_matches=True,
)
view.export(
    export_dir="data/coco_nav/train",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    classes=MY_CLASSES,
)
```

**Nota**: FiftyOne puede crear subdirectorios extra con el nombre del split
(ej: `images/val/` dentro de `train/`). Verificar la estructura antes de combinar.

### Paso 3 — Combinar datasets

- Copiar imagenes de ambas fuentes a `nav_combined/images/{train,val}/`
- Prefijos para evitar colisiones: `coco_*.jpg`, `custom_*.jpg`
- Labels de COCO: IDs ya correctos (0-N segun orden de clases)
- Labels custom: remapear IDs (ej: 0-5 → 18-23)

### Paso 4 — dataset.yaml

```yaml
path: /path/to/data/nav_combined
train: ./images/train/
val: ./images/val/
names:
  0: person        # COCO
  1: bicycle       # COCO
  ...
  17: potted plant  # COCO
  18: Door          # custom
  19: Stairs        # custom
  ...
  23: Wheelchair    # custom
```

### Paso 5 — Entrenar

```bash
uv run python scripts/train.py \
  --model yolo26s.pt \
  --data data/nav_combined/dataset.yaml \
  --epochs 50
```

Sin `freeze` — el modelo necesita aprender la nueva cabeza de deteccion
con todas las clases.

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

### Referencia de tiempos (RTX 5060 Ti 16GB)

| Dataset | Imagenes | Batch | Tiempo/epoch |
|---------|----------|-------|-------------|
| nav_combined (24 clases) | 11,608 | 11 | ~122s |
| nav_custom (6 clases) | 3,608 | 11 | ~31s |

## Herramientas

- **FiftyOne**: Descarga de COCO y Open Images V7 con filtros por clase
- **Ultralytics YOLO26**: Entrenamiento, export, inferencia
- **TensorRT**: Export FP16 para maxima velocidad
