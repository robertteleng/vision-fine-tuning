# HITOS del Proyecto - Framework 3x3

Fine-tuning YOLO12s para deteccin de pilares VR con patrn tablero amarillo-negro.

---

## HITO 1: Auto-Anotacin con Template Matching

**Descripcin:** Generar anotaciones YOLO automticamente usando template matching multi-escala.

**Goal:** Crear archivos `.txt` con bounding boxes en formato YOLO para cada frame.

**Inputs:**
- `data/templates/pillar.jpg` - Template del pilar (280x544 px)
- `data/video_frames/*.jpg` - Frames extrados del video

**Outputs:**
- `data/labels/*.txt` - Anotaciones en formato YOLO (clase x_center y_center width height)

**Minimap:**
```
                 +------------------+
                 |   load_template  |
                 +--------+---------+
                          |
                          v
+----------+    +-------------------+    +------------------+
| Template +--->| match_multiscale  +--->|  Lista boxes     |
| + Frame  |    | (20 escalas)      |    |  con confianza   |
+----------+    +-------------------+    +--------+---------+
                                                  |
                                                  v
                                         +-------+--------+
                                         |      NMS       |
                                         | (IoU < 0.3)    |
                                         +-------+--------+
                                                 |
                                                 v
                                         +-------+--------+
                                         | to_yolo_format |
                                         | (normalizar)   |
                                         +-------+--------+
                                                 |
                                                 v
                                         +-------+--------+
                                         |  frame_XXX.txt |
                                         +----------------+
```

**Breakdown 3x3:**
```
scripts/auto_annotate.py
|
+-- 1. load_template(path)
|       |-- cv2.imread(path, GRAYSCALE)
|       |-- Validar que existe
|       +-- Retornar array numpy
|
+-- 2. match_template_multiscale(image, template, scales, threshold)
|       |-- Para cada escala en [0.3, 0.4, ... 2.0]:
|       |       cv2.resize(template, scale)
|       |       cv2.matchTemplate(TM_CCOEFF_NORMED)
|       |       Guardar picos >= threshold
|       +-- Retornar lista [{x, y, w, h, conf}, ...]
|
+-- 3. nms(detections, iou_threshold=0.3)
        |-- Ordenar por confianza (mayor primero)
        |-- Para cada box: descartar si IoU > threshold con anteriores
        +-- Retornar boxes filtradas
```

---

## HITO 2: Divisin del Dataset

**Descripcin:** Dividir imgenes y labels en conjuntos train/val manteniendo pares consistentes.

**Goal:** Crear estructura de carpetas YOLO con split 80/20 reproducible.

**Inputs:**
- `data/video_frames/*.jpg` - Imgenes
- `data/labels/*.txt` - Labels YOLO

**Outputs:**
- `data/dataset/train/images/` + `labels/`
- `data/dataset/val/images/` + `labels/`

**Minimap:**
```
+-------------+     +-------------+
|   images/   |     |   labels/   |
| frame_*.jpg |     | frame_*.txt |
+------+------+     +------+------+
       |                   |
       +--------+----------+
                |
                v
        +-------+--------+
        | get_matching   |
        | _pairs()       |
        +-------+--------+
                |
                v
        +-------+--------+
        | split_pairs    |
        | (seed=42)      |
        | 80% / 20%      |
        +-------+--------+
                |
       +--------+--------+
       |                 |
       v                 v
+------+------+   +------+------+
|   train/    |   |    val/     |
| images/     |   | images/     |
| labels/     |   | labels/     |
+-------------+   +-------------+
```

**Breakdown 3x3:**
```
scripts/split_dataset.py
|
+-- 1. get_matching_pairs(images_dir, labels_dir)
|       |-- Listar *.jpg en images_dir
|       |-- Para cada imagen: buscar label correspondiente
|       +-- Retornar solo pares donde ambos existen
|
+-- 2. split_pairs(pairs, train_ratio=0.8, seed=42)
|       |-- random.seed(42) para reproducibilidad
|       |-- random.shuffle(pairs)
|       +-- split_idx = int(len * 0.8)
|
+-- 3. copy_files(pairs, output_dir, split_name)
        |-- Crear directorios train/ y val/
        |-- shutil.copy() cada imagen y label
        +-- Verificar integridad
```

---

## HITO 3: Visualizacin de Anotaciones

**Descripcin:** Dibujar bounding boxes sobre imgenes para verificar calidad de anotaciones.

**Goal:** Generar imgenes con boxes dibujados para revisin visual rpida.

**Inputs:**
- `data/dataset/*/images/*.jpg`
- `data/dataset/*/labels/*.txt`

**Outputs:**
- Imgenes con rectngulos verdes dibujados

**Minimap:**
```
+-------------+     +-------------+
|  imagen.jpg |     |  imagen.txt |
+------+------+     +------+------+
       |                   |
       v                   v
+------+------+     +------+--------+
| cv2.imread  |     | load_         |
|             |     | annotations() |
+------+------+     +------+--------+
       |                   |
       |    +--------------+
       |    |
       v    v
+------+----+------+
| yolo_to_pixels() |
| (desnormalizar)  |
+--------+---------+
         |
         v
+--------+---------+
| draw_annotations |
| cv2.rectangle()  |
| cv2.putText()    |
+--------+---------+
         |
         v
+--------+---------+
|  imagen_viz.jpg  |
+------------------+
```

**Breakdown 3x3:**
```
scripts/visualize_annotations.py
|
+-- 1. load_annotations(label_path)
|       |-- Leer .txt lnea por lnea
|       |-- Parsear: cls, x, y, w, h (floats)
|       +-- Retornar lista de dicts
|
+-- 2. yolo_to_pixels(ann, img_w, img_h)
|       |-- x1 = (x_center - w/2) * img_w
|       |-- y1 = (y_center - h/2) * img_h
|       +-- Retornar (x1, y1, x2, y2) en pxeles
|
+-- 3. draw_annotations(image, annotations)
        |-- Para cada ann: yolo_to_pixels()
        |-- cv2.rectangle(img, (x1,y1), (x2,y2), GREEN, 2)
        +-- cv2.putText(img, "class N", ...)
```

---

## HITO 4: Entrenamiento del Modelo

**Descripcin:** Fine-tuning de YOLO12s preentrenado en COCO para detectar pilares VR.

**Goal:** Modelo con mAP@50 > 95% y Recall > 95%.

**Inputs:**
- `yolo12s.pt` - Modelo base preentrenado
- `data/pillar.yaml` - Configuracin del dataset
- `config.yaml` - Hiperparmetros

**Outputs:**
- `models/yolo12s_pillars_YYYYMMDD.pt` - Modelo fine-tuned
- `runs/train/exp/` - Logs, mtricas, checkpoints

**Minimap:**
```
+-------------+     +-------------+     +-------------+
| yolo12s.pt  |     | pillar.yaml |     | config.yaml |
| (COCO)      |     | (dataset)   |     | (hparams)   |
+------+------+     +------+------+     +------+------+
       |                   |                   |
       +-------------------+-------------------+
                           |
                           v
                   +-------+--------+
                   | check_gpu()    |
                   | load_config()  |
                   +-------+--------+
                           |
                           v
                   +-------+--------+
                   | model.train()  |
                   | epochs=100     |
                   | batch=8        |
                   +-------+--------+
                           |
                           v
                   +-------+--------+
                   | Guardar con    |
                   | timestamp      |
                   +-------+--------+
                           |
              +------------+------------+
              |                         |
              v                         v
      +-------+-------+         +-------+-------+
      | best.pt       |         | runs/train/   |
      | (en models/)  |         | (logs)        |
      +---------------+         +---------------+
```

**Breakdown 3x3:**
```
scripts/train.py
|
+-- 1. check_gpu_availability() + load_config()
|       |-- torch.cuda.is_available()
|       |-- yaml.safe_load(config.yaml)
|       +-- Validar data/pillar.yaml existe
|
+-- 2. setup_training(config)
|       |-- model = YOLO("yolo12s.pt")
|       |-- Configurar augmentations (hsv, mosaic, scale)
|       +-- Preparar training_params dict
|
+-- 3. train_and_save(model, config)
        |-- results = model.train(data, epochs, batch, ...)
        |-- Copiar best.pt a models/ con timestamp
        +-- Mostrar mtricas finales
```

---

## HITO 5: Inferencia

**Descripcin:** Ejecutar deteccin en imgenes, videos o webcam usando el modelo entrenado.

**Goal:** Detectar pilares en tiempo real (~180 FPS con TensorRT).

**Inputs:**
- `models/*.pt` o `*.engine` - Modelo
- Imagen / Video / Webcam (source)

**Outputs:**
- Imagen/video con detecciones dibujadas
- Estadsticas (num detecciones, confianza, FPS)

**Minimap:**
```
                   +----------------+
                   | find_best_     |
                   | model()        |
                   +-------+--------+
                           |
                           v
+-------------+    +-------+--------+
|   source    +--->| model.predict  |
| (img/vid/0) |    | conf, iou, ... |
+-------------+    +-------+--------+
                           |
                           v
                   +-------+--------+
                   | results[0]     |
                   | .boxes         |
                   | .plot()        |
                   +-------+--------+
                           |
              +------------+------------+
              |                         |
              v                         v
      +-------+-------+         +-------+-------+
      | Imagen con    |         | Estadsticas  |
      | detecciones   |         | JSON/log      |
      +---------------+         +---------------+
```

**Breakdown 3x3:**
```
scripts/inference.py
|
+-- 1. find_best_model()
|       |-- Buscar en models/ (prioridad: .engine > .pt > .onnx)
|       |-- Ordenar por fecha (ms reciente primero)
|       +-- Retornar path o FileNotFoundError
|
+-- 2. run_inference(source, model, conf, iou)
|       |-- model = YOLO(model_path)
|       |-- results = model(source, conf=0.65, iou=0.45)
|       +-- Extraer boxes, confs, imagen anotada
|
+-- 3. save_results(results, output_dir)
        |-- Imagen: cv2.imwrite()
        |-- Video: VideoWriter frame por frame
        +-- Log estadsticas
```

---

## HITO 6: Auto-Anotacin Zero-Shot con Grounding DINO

**Descripcin:** Anotar imgenes usando descripcin en lenguaje natural sin entrenamiento previo.

**Goal:** Crear dataset YOLO completo desde prompt de texto ("yellow and black checkered pillar").

**Inputs:**
- `data/video_frames/*.jpg` - Imgenes sin anotar
- Prompt: "yellow and black checkered pillar"

**Outputs:**
- `data/dataset_auto/` - Dataset completo (images/, labels/, dataset.yaml)

**Minimap:**
```
+-------------+     +------------------+
| "yellow and |     | Grounding DINO   |
| black..."   |     | (HuggingFace)    |
+------+------+     +--------+---------+
       |                     |
       +----------+----------+
                  |
                  v
          +-------+--------+
          | AutoProcessor   |
          | from_pretrained |
          +-------+--------+
                  |
                  v
+----------+     +-------+--------+
|  imagen  +---->| model(**inputs)|
+----------+     | threshold=0.3  |
                 +-------+--------+
                         |
                         v
                 +-------+--------+
                 | post_process   |
                 | boxes, scores  |
                 +-------+--------+
                         |
            +------------+------------+
            |                         |
            v                         v
    +-------+-------+         +-------+-------+
    |  labels/      |         |  images/      |
    |  *.txt        |         |  train/val    |
    +---------------+         +---------------+
```

**Breakdown 3x3:**
```
scripts/auto_annotate_grounding_dino.py
|
+-- 1. load_grounding_dino(model_name)
|       |-- processor = AutoProcessor.from_pretrained()
|       |-- model = AutoModelForZeroShotObjectDetection()
|       +-- Mover a GPU si disponible (.to(device))
|
+-- 2. detect_with_prompt(image, prompt, threshold)
|       |-- inputs = processor(image, text=prompt+".")
|       |-- outputs = model(**inputs)
|       +-- Filtrar boxes donde score >= threshold
|
+-- 3. create_yolo_dataset(source, output, prompt)
        |-- Procesar todas las imgenes con tqdm
        |-- Dividir train/val (80/20, seed=42)
        +-- Generar dataset.yaml con rutas absolutas
```

---

## HITO 7: Testing Completo

**Descripcin:** Suite de tests automatizados para validar todas las funcionalidades.

**Goal:** 100% de tests pasando, cobertura de funciones crticas.

**Inputs:**
- Cdigo fuente en `scripts/`
- Dataset en `data/dataset/`

**Outputs:**
- `tests/` - 47 tests organizados
- `pytest.ini` - Configuracin

**Minimap:**
```
+-------------+     +-------------+     +-------------+
|   scripts/  |     |    data/    |     |   models/   |
|   *.py      |     |  dataset/   |     |   *.pt      |
+------+------+     +------+------+     +------+------+
       |                   |                   |
       +-------------------+-------------------+
                           |
                           v
                   +-------+--------+
                   |    pytest      |
                   | tests/ -v      |
                   +-------+--------+
                           |
       +-------------------+-------------------+
       |                   |                   |
       v                   v                   v
+------+------+    +-------+-------+   +------+------+
| Unitarios   |    | Integracin   |   | Fixtures    |
| (funciones) |    | (pipelines)   |   | (conftest)  |
+-------------+    +---------------+   +-------------+
       |                   |
       v                   v
+------+------------------+-------+
|      47 tests PASSED           |
+--------------------------------+
```

**Breakdown 3x3:**
```
tests/
|
+-- 1. Tests Unitarios
|       |-- test_auto_annotate.py (NMS, YOLO format, template)
|       |-- test_split_dataset.py (pairs, split ratio, seed)
|       +-- test_visualize.py (load_annotations, yolo_to_pixels)
|
+-- 2. Tests de Integracin
|       |-- test_integration.py (config existe, models existen)
|       |-- test_data_integrity.py (estructura dirs, formato labels)
|       +-- test_inference.py (modelo carga, predice)
|
+-- 3. Configuracin
        |-- conftest.py (fixtures: sample_image, sample_label)
        |-- pytest.ini (testpaths, markers)
        +-- Ejecutar: pytest tests/ -v
```

---

## HITO 8: GUI Robusta con Gradio

**Descripcin:** Aplicacin web Gradio con todas las funcionalidades del proyecto.

**Goal:** Interfaz completa con 7 tabs funcionales y buen error handling.

**Inputs:**
- Modelos en `models/`
- Dataset en `data/dataset/`
- Configuracin en `config.yaml`

**Outputs:**
- Servidor web en `localhost:7860`
- 7 tabs: Inference, Metrics, Training, Auto-Annotate, Dataset, Annotations, Info

**Minimap:**
```
                   +----------------+
                   |    app.py      |
                   |   create_app() |
                   +-------+--------+
                           |
                           v
                   +-------+--------+
                   |  gr.Blocks()   |
                   |  gr.Tabs()     |
                   +-------+--------+
                           |
    +----------+-----------+-----------+-----------+----------+----------+
    |          |           |           |           |          |          |
    v          v           v           v           v          v          v
+---+---+ +----+----+ +----+----+ +----+----+ +----+----+ +---+----+ +---+---+
|Infer- | |Metrics  | |Training | |Auto-    | |Dataset  | |Annot-  | | Info  |
|ence   | |         | |         | |Annotate | |Manager  | |ations  | |       |
+---+---+ +----+----+ +----+----+ +----+----+ +----+----+ +---+----+ +---+---+
    |          |           |           |           |          |
    v          v           v           v           v          v
  Image    Benchmark    Epochs      GDino      Stats      Review
  Video    mAP/FPS     Batch       Single     Validate   Edit
                       Model       Batch      Export     Delete
```

**Breakdown 3x3:**
```
app.py
|
+-- 1. Tabs de Uso Principal (Inference, Metrics, Training)
|       |-- run_inference_image(img, model, conf, iou)
|       |-- run_inference_video(video, model, conf, iou)
|       |-- get_training_metrics(), run_benchmark()
|       +-- start_training(epochs, batch, model_base)
|
+-- 2. Tabs de Gestin de Datos (Auto-Annotate, Dataset, Annotations)
|       |-- AutoAnnotator class (load_model, annotate_image, annotate_folder)
|       |-- get_dataset_stats(), validate_dataset(), export_dataset()
|       +-- AnnotationReviewer class (load, next, prev, delete)
|
+-- 3. UX y Error Handling
        |-- Mensajes con emojis ( )
        |-- Validacin de parmetros (0 < conf <= 1)
        +-- Manejo especfico (CUDA OOM, file not found)
```

---

## Resumen

| HITO | Descripcin | Script | Estado |
|------|-------------|--------|--------|
| 1 | Auto-anotacin Template Matching | `auto_annotate.py` |  |
| 2 | Divisin del Dataset | `split_dataset.py` |  |
| 3 | Visualizacin de Anotaciones | `visualize_annotations.py` |  |
| 4 | Entrenamiento YOLO12s | `train.py` |  |
| 5 | Inferencia | `inference.py` |  |
| 6 | Auto-anotacin Grounding DINO | `auto_annotate_grounding_dino.py` |  |
| 7 | Testing Completo | `tests/` |  |
| 8 | GUI Robusta | `app.py` |  |

---

## Mtricas Finales

| Mtrica | Valor |
|---------|-------|
| mAP@50 | 98.7% |
| mAP@50-95 | 87.4% |
| Precisin | 91.7% |
| Recall | 99.2% |
| Inferencia TensorRT | 5.5ms (180 FPS) |
| Tests | 47 passing |

---

*ltima actualizacin: Diciembre 2025*
