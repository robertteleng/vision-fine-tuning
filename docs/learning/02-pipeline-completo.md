# Pipeline Completo: De Cero a Produccion

## Diagrama General

```
Imagenes/Video → Grounding DINO (auto-anotacion) → Dataset YOLO → Fine-tune YOLO26m → TensorRT FP16 → Inferencia
```

## Pasos Detallados

### Paso 1: Recopilar Imagenes

Necesitas imagenes de los objetos que quieres detectar. Opciones:

```bash
# Opcion A: Extraer frames de video
ffmpeg -i video_calle.mp4 -vf "fps=2" data/frames/frame_%05d.jpg

# Opcion B: Descargar de internet (para prototipo)
# Buscar en Google Images, Flickr, etc.

# Opcion C: Grabar con movil/webcam
python scripts/inference.py --source 0 --save-frames data/frames/
```

**Requisitos minimos:**
- 100-500 imagenes por clase
- Variedad de angulos, distancias, iluminacion
- Objetos visibles (>32px en la imagen)
- Formato: JPG o PNG

**Estructura esperada:**
```
data/
└── frames/
    ├── img_001.jpg
    ├── img_002.jpg
    └── ...
```

### Paso 2: Auto-Anotacion con Grounding DINO

Grounding DINO anota automaticamente usando descripcion textual. No necesitas etiquetar nada a mano.

**Clase unica:**
```bash
python scripts/auto_annotate_grounding_dino.py \
    --source data/frames/ \
    --prompt "staircase steps" \
    --output data/dataset_stairs/ \
    --threshold 0.3
```

**Multi-clase (ejecutar una vez por clase):**
```bash
# Clase 0: escaleras
python scripts/auto_annotate_grounding_dino.py \
    --source data/frames/ \
    --prompt "staircase steps" \
    --output data/dataset_nav/ \
    --threshold 0.3 \
    --class-name stairs

# Clase 1: puertas
python scripts/auto_annotate_grounding_dino.py \
    --source data/frames/ \
    --prompt "door entrance" \
    --output data/dataset_nav/ \
    --threshold 0.3 \
    --class-name door

# Clase 2: paso de peatones
python scripts/auto_annotate_grounding_dino.py \
    --source data/frames/ \
    --prompt "pedestrian crosswalk zebra crossing" \
    --output data/dataset_nav/ \
    --threshold 0.35 \
    --class-name crosswalk
```

**Que genera:**
```
data/dataset_nav/
├── images/
│   ├── train/        # 80% de imagenes
│   └── val/          # 20% de imagenes
├── labels/
│   ├── train/        # Labels YOLO (.txt)
│   └── val/
├── visualizations/   # Previews con bounding boxes
└── dataset.yaml      # Config para entrenamiento
```

**Formato de label YOLO (cada .txt):**
```
0 0.523456 0.345678 0.234567 0.456789
1 0.712345 0.567890 0.123456 0.234567
```
Donde: `clase x_centro y_centro ancho alto` (todo normalizado 0-1).

### Paso 3: Revisar Anotaciones

Antes de entrenar, revisa que las anotaciones son correctas:

```bash
# Visualizar anotaciones
python scripts/visualize_annotations.py --dataset data/dataset_nav/

# O usar la UI web
python app.py
# Tab "Annotations" → Cargar dataset → Navegar imagenes
```

**Que buscar:**
- Bounding boxes bien centrados en los objetos
- Sin falsos positivos evidentes (detecta cosas que no son)
- Cobertura razonable (>70% de objetos detectados)

**Si hay errores:**
- Ajustar `--threshold` (subir si muchos falsos positivos, bajar si no detecta)
- Cambiar el prompt (mas especifico o sinonimos)
- Borrar labels malos manualmente o con la UI

### Paso 4: Editar dataset.yaml (si multi-clase)

Si anotaste multiples clases por separado, necesitas un `dataset.yaml` unificado:

```yaml
# data/dataset_nav/dataset.yaml
path: ~/Projects/ml/fine-tuning/data/dataset_nav
train: images/train
val: images/val

names:
  0: stairs
  1: door
  2: crosswalk
  3: curb
  4: pole
```

### Paso 5: Fine-Tune YOLO26m

```bash
# Entrenamiento con config por defecto
python scripts/train.py --data data/dataset_nav/dataset.yaml

# Con overrides
python scripts/train.py \
    --data data/dataset_nav/dataset.yaml \
    --epochs 100 \
    --batch -1 \
    --name nav_detector_v1
```

**Que pasa internamente:**
1. Carga `yolo26m.pt` (modelo pre-entrenado en COCO)
2. Auto-detecta GPU y ajusta batch size
3. Entrena con augmentacion (mosaic, flip, hsv, scale)
4. Guarda checkpoints cada 10 epocas
5. Early stopping si no mejora en 20 epocas
6. Guarda `best.pt` en `models/`

**Monitorizar entrenamiento:**
```bash
# Ver metricas en tiempo real
python app.py
# Tab "Metrics" → Seleccionar run
```

**Resultados esperados (100 epocas, ~500 imagenes):**
- mAP@50: 85-95%
- mAP@50-95: 60-80%
- Tiempo: 30-60 min en RTX 5060 Ti

### Paso 6: Evaluar Modelo

```bash
python scripts/evaluate.py \
    --model models/yolo26m_20260220.pt \
    --data data/dataset_nav/dataset.yaml
```

Genera metricas por clase: precision, recall, mAP50, mAP50-95, F1.

### Paso 7: Exportar a TensorRT

```bash
# TensorRT FP16 (maximo rendimiento en NVIDIA)
python scripts/export_tensorrt.py --format engine --half

# ONNX (portable a cualquier plataforma)
python scripts/export_tensorrt.py --format onnx
```

### Paso 8: Benchmark Final

```bash
python scripts/benchmark.py --model models/yolo26m.pt --iterations 200
```

Compara PyTorch vs ONNX vs TensorRT automaticamente.

### Paso 9: Inferencia en Produccion

```bash
# Imagen
python scripts/inference.py --source foto_calle.jpg

# Video
python scripts/inference.py --source video_navegacion.mp4

# Webcam en tiempo real
python scripts/inference.py --source 0 --show

# Con modelo TensorRT especifico
python scripts/inference.py --source 0 --model models/yolo26m.engine --show
```

## Resumen del Pipeline

| Paso | Herramienta | Tiempo estimado |
|------|-------------|-----------------|
| 1. Recopilar imagenes | ffmpeg / movil | 1-2 horas |
| 2. Auto-anotar | Grounding DINO | 5-30 min (GPU) |
| 3. Revisar | UI Gradio | 15-30 min |
| 4. Config dataset | Editor de texto | 5 min |
| 5. Fine-tune | YOLO26m | 30-60 min (GPU) |
| 6. Evaluar | evaluate.py | 2 min |
| 7. Exportar TensorRT | export_tensorrt.py | 30 seg |
| 8. Benchmark | benchmark.py | 1 min |
| 9. Inferencia | inference.py | Tiempo real |

**Total: ~3-4 horas de cero a modelo en produccion.**
