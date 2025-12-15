# HITOS del Proyecto - Framework 3x3

Documentación de los 8 HITOs del proyecto de fine-tuning YOLO12s para detección de pilares VR.

---

## HITO 1: Auto-Anotación con Template Matching

**Descripción:** Generar anotaciones YOLO automáticamente usando template matching multi-escala.

**Goal:** Crear archivos `.txt` con bounding boxes en formato YOLO para cada frame.

**Inputs:**
- `data/templates/pillar.jpg` - Template del pilar
- `data/video_frames/*.jpg` - Frames extraídos del video

**Outputs:**
- `data/labels/*.txt` - Anotaciones en formato YOLO

**Minimap:**
```
[Template + Frames] → detect_boxes() → NMS → [Labels YOLO]
```

**Breakdown 3x3:**
```
scripts/auto_annotate.py
│
├── 1. load_resources()
│   ├── cv2.imread(template_path)
│   ├── glob.glob(frames_dir/*.jpg)
│   └── Validar que existan archivos
│
├── 2. detect_boxes(frame, template, scales, threshold)
│   ├── matchTemplate() para cada escala 0.3→2.0
│   ├── Encontrar picos > threshold
│   └── Convertir a [x, y, w, h, conf]
│
└── 3. save_annotations(detections, output_path)
    ├── NMS para eliminar duplicados
    ├── Convertir a YOLO normalizado
    └── Escribir archivo .txt
```

---

## HITO 2: División del Dataset

**Descripción:** Dividir imágenes y labels en conjuntos train/val manteniendo pares consistentes.

**Goal:** Crear estructura de carpetas YOLO con split 80/20.

**Inputs:**
- `data/video_frames/*.jpg` - Imágenes
- `data/labels/*.txt` - Labels

**Outputs:**
- `data/dataset/train/images/` + `labels/`
- `data/dataset/val/images/` + `labels/`

**Minimap:**
```
[Imágenes + Labels] → emparejar → shuffle → split 80/20 → [Dataset YOLO]
```

**Breakdown 3x3:**
```
scripts/split_dataset.py
│
├── 1. get_matching_pairs(images_dir, labels_dir)
│   ├── Listar imágenes (*.jpg, *.png)
│   ├── Buscar label correspondiente
│   └── Retornar solo pares válidos
│
├── 2. split_pairs(pairs, val_ratio, seed)
│   ├── random.seed(42) reproducibilidad
│   ├── random.shuffle(pairs)
│   └── Dividir por ratio
│
└── 3. copy_to_dataset(pairs, split_name, output_dir)
    ├── Crear directorios train/val
    ├── Copiar archivos
    └── Verificar integridad
```

---

## HITO 3: Visualización de Anotaciones

**Descripción:** Dibujar bounding boxes sobre imágenes para verificar calidad de anotaciones.

**Goal:** Generar imágenes con boxes dibujados para revisión visual.

**Inputs:**
- `data/dataset/*/images/*.jpg`
- `data/dataset/*/labels/*.txt`

**Outputs:**
- Imágenes con rectángulos dibujados (pantalla o archivo)

**Minimap:**
```
[Imagen + Label] → parsear YOLO → convertir a píxeles → [Imagen con boxes]
```

**Breakdown 3x3:**
```
scripts/visualize_annotations.py
│
├── 1. load_annotations(label_path)
│   ├── Leer .txt línea por línea
│   ├── Parsear: cls, x, y, w, h
│   └── Retornar lista de dicts
│
├── 2. yolo_to_pixels(ann, img_w, img_h)
│   ├── x1 = (x_center - w/2) * img_w
│   ├── y1 = (y_center - h/2) * img_h
│   └── Calcular x2, y2
│
└── 3. draw_boxes(image, annotations)
    ├── Para cada anotación: yolo_to_pixels()
    ├── cv2.rectangle()
    └── cv2.putText() con clase
```

---

## HITO 4: Entrenamiento del Modelo

**Descripción:** Fine-tuning de YOLO12s preentrenado en COCO para detectar pilares VR.

**Goal:** Modelo entrenado con mAP@50 > 95% y Recall > 95%.

**Inputs:**
- `yolo12s.pt` - Modelo base preentrenado
- `data/pillar.yaml` - Configuración del dataset
- `config.yaml` - Hiperparámetros

**Outputs:**
- `models/yolo12s_pillars_YYYYMMDD.pt` - Modelo entrenado
- `runs/train/exp/` - Logs y métricas

**Minimap:**
```
[YOLO12s + Dataset] → train(100 epochs) → evaluar → [Modelo fine-tuned]
```

**Breakdown 3x3:**
```
scripts/train.py
│
├── 1. load_config()
│   ├── Leer config.yaml (epochs, batch, imgsz)
│   ├── Validar data/pillar.yaml existe
│   └── Verificar estructura dataset
│
├── 2. setup_training(config)
│   ├── model = YOLO("yolo12s.pt")
│   ├── Configurar augmentations (hsv, translate, scale, mosaic)
│   └── Configurar callbacks/logging
│
└── 3. train_and_save(model, config)
    ├── model.train(data, epochs, batch, imgsz)
    ├── Evaluar métricas finales
    └── Copiar best.pt a models/
```

---

## HITO 5: Inferencia

**Descripción:** Ejecutar detección en imágenes, videos o webcam usando el modelo entrenado.

**Goal:** Detectar pilares en tiempo real (~180 FPS con TensorRT).

**Inputs:**
- `models/*.pt` o `*.engine` - Modelo
- Imagen / Video / Stream de webcam

**Outputs:**
- Imagen/video con detecciones dibujadas
- Estadísticas (num detecciones, confianza, FPS)

**Minimap:**
```
[Modelo + Source] → predict() → plot() → [Resultado anotado]
```

**Breakdown 3x3:**
```
scripts/inference.py
│
├── 1. find_best_model(models_dir)
│   ├── Buscar por prioridad: .engine > .pt > .onnx
│   ├── Ordenar por fecha
│   └── Retornar más reciente
│
├── 2. run_inference(source, model, conf, iou)
│   ├── model = YOLO(model_path)
│   ├── results = model(source, conf, iou)
│   └── Extraer boxes, confs, annotated
│
└── 3. save_results(results, output_dir)
    ├── Imagen: cv2.imwrite()
    ├── Video: VideoWriter frame por frame
    └── Generar estadísticas
```

---

## HITO 6: Auto-Anotación Zero-Shot con Grounding DINO

**Descripción:** Anotar imágenes usando descripción en lenguaje natural sin entrenamiento previo.

**Goal:** Crear dataset YOLO completo desde prompt de texto.

**Inputs:**
- `data/video_frames/*.jpg` - Imágenes sin anotar
- Prompt: "yellow and black checkered pillar"

**Outputs:**
- `data/dataset_auto/` - Dataset completo (images/, labels/, dataset.yaml)

**Minimap:**
```
[Imágenes + Prompt] → Grounding DINO → filtrar threshold → [Dataset YOLO]
```

**Breakdown 3x3:**
```
scripts/auto_annotate_grounding_dino.py
│
├── 1. load_grounding_dino(model_name)
│   ├── AutoProcessor.from_pretrained()
│   ├── AutoModelForZeroShotObjectDetection.from_pretrained()
│   └── Mover a GPU si disponible
│
├── 2. detect_with_prompt(image, prompt, threshold)
│   ├── inputs = processor(image, text=prompt+".")
│   ├── outputs = model(**inputs)
│   └── Filtrar scores >= threshold
│
└── 3. create_yolo_dataset(source_dir, output_dir, prompt)
    ├── Procesar todas las imágenes
    ├── Dividir train/val automáticamente
    └── Generar dataset.yaml
```

---

## HITO 7: Testing Completo

**Descripción:** Suite de tests automatizados para validar todas las funcionalidades.

**Goal:** 100% de tests pasando, cobertura de funciones críticas.

**Inputs:**
- Código fuente en `scripts/`
- Dataset en `data/dataset/`

**Outputs:**
- `tests/` - 47 tests organizados
- `pytest.ini` - Configuración

**Minimap:**
```
[Código + Datos] → pytest → [47 tests ✅]
```

**Breakdown 3x3:**
```
tests/
│
├── 1. Tests Unitarios
│   ├── test_auto_annotate.py (NMS, YOLO format, template)
│   ├── test_split_dataset.py (pairs, split ratio)
│   └── test_visualize.py (load_annotations, yolo_to_pixels)
│
├── 2. Tests de Integración
│   ├── test_integration.py (config, models, inference)
│   ├── test_data_integrity.py (estructura, formato labels)
│   └── conftest.py (fixtures compartidos)
│
└── 3. Configuración
    ├── pytest.ini (testpaths, addopts)
    ├── Ejecutar: pytest tests/ -v
    └── Resultado: 47 passed
```

---

## HITO 8: GUI Robusta

**Descripción:** Aplicación web Gradio con todas las funcionalidades del proyecto.

**Goal:** Interfaz completa con 7 tabs funcionales y buen error handling.

**Inputs:**
- Modelos en `models/`
- Dataset en `data/dataset/`
- Configuración en `config.yaml`

**Outputs:**
- Servidor web en `localhost:7860`
- 7 tabs: Inference, Metrics, Training, Auto-Annotate, Dataset, Annotations, Info

**Minimap:**
```
[Gradio Blocks] → 7 Tabs → callbacks → [App Web]
```

**Breakdown 3x3:**
```
app.py
│
├── 1. Tabs de Uso Principal
│   ├── Inference (Image/Video con sliders conf/iou)
│   ├── Metrics (métricas + benchmark)
│   └── Training (epochs, batch, modelo base)
│
├── 2. Tabs de Gestión de Datos
│   ├── Auto-Annotate (Grounding DINO single/batch)
│   ├── Dataset (estadísticas, validación, export)
│   └── Annotations (revisor interactivo)
│
└── 3. UX y Error Handling
    ├── Mensajes con emojis (✅ ⚠️ ❌)
    ├── Validación de parámetros
    └── Manejo específico (CUDA OOM, file not found)
```

---

## Resumen

| HITO | Descripción | Script | Estado |
|------|-------------|--------|--------|
| 1 | Auto-anotación Template Matching | `auto_annotate.py` | ✅ |
| 2 | División del Dataset | `split_dataset.py` | ✅ |
| 3 | Visualización de Anotaciones | `visualize_annotations.py` | ✅ |
| 4 | Entrenamiento YOLO12s | `train.py` | ✅ |
| 5 | Inferencia | `inference.py` | ✅ |
| 6 | Auto-anotación Grounding DINO | `auto_annotate_grounding_dino.py` | ✅ |
| 7 | Testing Completo | `tests/` | ✅ |
| 8 | GUI Robusta | `app.py` | ✅ |

---

## Métricas Finales

| Métrica | Valor |
|---------|-------|
| mAP@50 | 98.7% |
| mAP@50-95 | 87.4% |
| Precisión | 91.7% |
| Recall | 99.2% |
| Inferencia TensorRT | 5.5ms (180 FPS) |
| Tests | 47 passing |

---

*Última actualización: Diciembre 2025*
