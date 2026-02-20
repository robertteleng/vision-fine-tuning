# HITOS del Proyecto - Framework 3x3

Fine-tuning YOLO12s para detección de pilares VR con patrón tablero amarillo-negro.

---

## HITO 1: Auto-Anotación con Template Matching

**Descripción:** Generar anotaciones YOLO automáticamente usando template matching multi-escala.

**Goal:** Crear archivos `.txt` con bounding boxes en formato YOLO para cada frame.

**Inputs:**
- `data/templates/pillar.jpg` - Template del pilar (280x544 px)
- `data/video_frames/*.jpg` - Frames extraídos del video

**Outputs:**
- `data/labels/*.txt` - Anotaciones en formato YOLO

**Minimap:**
```
Template + Frame --> match_multiscale (20 escalas) --> NMS --> to_yolo_format --> .txt
```

**Breakdown 3x3:**
```
scripts/auto_annotate.py
│
├── 1. load_template(path)
│       cv2.imread(path, GRAYSCALE)
│       # Carga template en escala de grises para matching
│
├── 2. match_template_multiscale(image, template, threshold=0.65)
│       for scale in np.linspace(0.3, 2.0, 20):    # 20 escalas
│           cv2.matchTemplate(TM_CCOEFF_NORMED)    # correlación normalizada
│           np.where(result >= threshold)          # picos sobre umbral
│       # Retorna lista de {x, y, w, h, conf}
│
└── 3. nms(detections, iou_threshold=0.3)
        scores.argsort()[::-1]                     # ordenar por confianza
        # Descartar si IoU > 0.3 con boxes ya aceptadas
        # Retorna boxes filtradas sin duplicados
```

---

## HITO 2: División del Dataset

**Descripción:** Dividir imágenes y labels en conjuntos train/val manteniendo pares consistentes.

**Goal:** Crear estructura de carpetas YOLO con split 80/20 reproducible.

**Inputs:**
- `data/video_frames/*.jpg`
- `data/labels/*.txt`

**Outputs:**
- `data/dataset/train/` y `val/` (images + labels)

**Minimap:**
```
images/ + labels/ --> emparejar --> shuffle(seed=42) --> 80/20 --> train/ + val/
```

**Breakdown 3x3:**
```
scripts/split_dataset.py
│
├── 1. get_matching_pairs(images_dir, labels_dir)
│       for img in glob('*.jpg'):
│           label = labels_dir / f"{img.stem}.txt"
│       # Solo retorna pares donde ambos archivos existen
│
├── 2. split_pairs(pairs, ratio=0.8, seed=42)
│       random.seed(42)                  # reproducibilidad
│       random.shuffle(pairs)
│       split_idx = int(len * 0.8)       # 80% train, 20% val
│
└── 3. copy_files(pairs, output_dir, split_name)
        shutil.copy(image, output/split/images/)
        shutil.copy(label, output/split/labels/)
```

---

## HITO 3: Visualización de Anotaciones

**Descripción:** Dibujar bounding boxes sobre imágenes para verificar calidad de anotaciones.

**Goal:** Generar imágenes con boxes dibujados para revisión visual rápida.

**Inputs:**
- `data/dataset/*/images/*.jpg`
- `data/dataset/*/labels/*.txt`

**Outputs:**
- Imágenes con rectángulos verdes dibujados

**Minimap:**
```
imagen + label.txt --> yolo_to_pixels --> cv2.rectangle --> imagen_viz.jpg
```

**Breakdown 3x3:**
```
scripts/visualize_annotations.py
│
├── 1. load_annotations(label_path)
│       line.split() --> cls, x, y, w, h
│       # Parsea formato YOLO: clase + coords normalizadas
│
├── 2. yolo_to_pixels(ann, img_w, img_h)
│       x1 = (x_center - w/2) * img_w    # desnormalizar
│       y1 = (y_center - h/2) * img_h
│       # Convierte [0,1] a píxeles absolutos
│
└── 3. draw_annotations(image, annotations)
        cv2.rectangle(img, (x1,y1), (x2,y2), GREEN, 2)
        cv2.putText(img, f"class {cls}", ...)
```

---

## HITO 4: Entrenamiento del Modelo

**Descripción:** Fine-tuning de YOLO12s preentrenado en COCO para detectar pilares VR.

**Goal:** Modelo con mAP@50 > 95% y Recall > 95%.

**Inputs:**
- `yolo12s.pt` - Modelo base preentrenado
- `data/pillar.yaml` - Configuración del dataset
- `config.yaml` - Hiperparámetros

**Outputs:**
- `models/yolo12s_pillars_YYYYMMDD.pt`
- `runs/train/exp/` - Logs y métricas

**Minimap:**
```
yolo12s.pt + pillar.yaml + config --> model.train(100 epochs) --> best.pt
```

**Breakdown 3x3:**
```
scripts/train.py
│
├── 1. check_gpu() + load_config()
│       torch.cuda.is_available()        # verificar GPU
│       yaml.safe_load(config.yaml)      # cargar hiperparámetros
│       # RTX 2060: batch=8, epochs=100
│
├── 2. setup_training(config)
│       model = YOLO('yolo12s.pt')       # cargar base preentrenada
│       params = {epochs, batch, imgsz=640, amp=True}
│       # amp=True para mixed precision (ahorra VRAM)
│
└── 3. train_and_save()
        model.train(data='pillar.yaml', **params)
        shutil.copy(best.pt, f"models/yolo12s_pillars_{timestamp}.pt")
        # Guarda con timestamp para versionado
```

---

## HITO 5: Inferencia

**Descripción:** Ejecutar detección en imágenes, videos o webcam usando el modelo entrenado.

**Goal:** Detectar pilares en tiempo real (~180 FPS con TensorRT).

**Inputs:**
- `models/*.pt` o `*.engine`
- Imagen / Video / Webcam

**Outputs:**
- Imagen/video con detecciones dibujadas
- Estadísticas (detecciones, confianza, FPS)

**Minimap:**
```
find_best_model() --> model.predict(source, conf=0.65) --> results[0].plot() --> output
```

**Breakdown 3x3:**
```
scripts/inference.py
│
├── 1. find_best_model()
│       glob('models/vr_boxes_*.pt')     # buscar modelos versionados
│       # Prioridad: .engine > .pt > .onnx
│       # Retorna el más reciente
│
├── 2. run_inference(source, conf=0.65, iou=0.45)
│       model = YOLO(model_path)
│       results = model.predict(source, conf, iou, stream=True)
│       # stream=True para procesar frame a frame
│
└── 3. process_results(results)
        boxes = result.boxes
        conf = box.conf[0]               # confianza de cada detección
        x1,y1,x2,y2 = box.xyxy[0]        # coordenadas del box
        annotated = result.plot()        # imagen con boxes dibujados
```

---

## HITO 6: Auto-Anotación Zero-Shot con Grounding DINO

**Descripción:** Anotar imágenes usando descripción en lenguaje natural sin entrenamiento previo.

**Goal:** Crear dataset YOLO completo desde prompt de texto.

**Inputs:**
- `data/video_frames/*.jpg`
- Prompt: `"yellow and black checkered pillar"`

**Outputs:**
- `data/dataset_auto/` (images/, labels/, dataset.yaml)

**Minimap:**
```
prompt + imagen --> Grounding DINO --> post_process(threshold=0.3) --> YOLO labels
```

**Breakdown 3x3:**
```
scripts/auto_annotate_grounding_dino.py
│
├── 1. load_grounding_dino()
│       AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
│       AutoModelForZeroShotObjectDetection.from_pretrained(...)
│       model.to("cuda")                 # mover a GPU
│
├── 2. detect_with_prompt(image, prompt, threshold=0.3)
│       prompt = prompt + "."            # DINO requiere punto final
│       inputs = processor(image, text=prompt)
│       results = processor.post_process_grounded_object_detection(...)
│       mask = scores >= threshold       # filtrar por confianza
│
└── 3. create_yolo_dataset(source, output, val_split=0.2)
        random.seed(42)                  # split reproducible
        # Para cada imagen: detectar + guardar label YOLO
        yaml.dump({path, train, val, names})  # generar dataset.yaml
```

---

## HITO 7: Testing Completo

**Descripción:** Suite de tests automatizados para validar todas las funcionalidades.

**Goal:** 100% de tests pasando, cobertura de funciones críticas.

**Inputs:**
- Código fuente en `scripts/`
- Dataset en `data/dataset/`

**Outputs:**
- `tests/` - 47 tests
- `pytest.ini`

**Minimap:**
```
scripts/*.py + data/ --> pytest tests/ -v --> 47 passed ✓
```

**Breakdown 3x3:**
```
tests/
│
├── 1. Tests Unitarios
│       test_nms_removes_overlapping()   # NMS elimina duplicados
│       test_yolo_format_normalized()    # valores en [0,1]
│       test_split_ratio()               # 80/20 correcto
│
├── 2. Tests de Integración
│       test_config_exists()             # config.yaml presente
│       test_dataset_structure()         # train/val/images/labels
│       test_model_loads()               # modelo carga sin error
│
└── 3. Fixtures (conftest.py)
        @pytest.fixture
        def sample_image():              # imagen dummy para tests
        def sample_label():              # label YOLO válido
```

---

## HITO 8: GUI Robusta con Gradio

**Descripción:** Aplicación web Gradio con arquitectura modular.

**Goal:** Interfaz completa con 7 tabs funcionales y lógica desacoplada en `src/`.

**Inputs:**
- `models/`, `data/dataset/`, `config.yaml`

**Outputs:**
- Servidor web en `localhost:7860`
- 7 tabs: Inference, Metrics, Training, Auto-Annotate, Dataset, Annotations, Info

**Minimap:**
```
src/*.py (lógica) --> app.py (UI Gradio) --> 7 tabs --> app.launch(port=7860)
```

**Breakdown 3x3:**
```
src/                              # Módulos de lógica de negocio
├── inference.py                  # find_available_models, run_inference_image/video
├── training.py                   # start_training, get_training_status
├── dataset.py                    # get_dataset_stats, validate_dataset, export
├── benchmark.py                  # run_benchmark
└── annotation.py                 # AutoAnnotator, AnnotationReviewer

app.py                            # Interfaz Gradio (ligera)
│
├── 1. Imports de src/
│       from src.inference import run_inference_image, ...
│       from src.annotation import AutoAnnotator, ...
│
├── 2. Wrappers para Gradio
│       def annotate_single_image(img, prompt, threshold):
│           return auto_annotator.annotate_image(...)
│       # Adapta clases/funciones de src/ a callbacks de Gradio
│
└── 3. create_app() --> gr.Blocks()
        with gr.Tabs():
            Tab Inference, Metrics, Training, Auto-Annotate, ...
        # UI pura, sin lógica de negocio
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
| 8 | GUI Robusta (modular) | `app.py` + `src/` | ✅ |

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

*Última actualización: Enero 2026*
