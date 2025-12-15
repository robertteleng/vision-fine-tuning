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
                                         +----------------+
```

**Breakdown 3x3:**
```
scripts/auto_annotate.py
│
├── 1. load_template(template_path)
│       template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
│       if template is None: raise FileNotFoundError
│       return template
│
├── 2. match_template_multiscale(image, template, scales, threshold=0.65)
│       for scale in np.linspace(0.3, 2.0, 20):
│           scaled = cv2.resize(template, (w*scale, h*scale))
│           result = cv2.matchTemplate(image, scaled, cv2.TM_CCOEFF_NORMED)
│           locations = np.where(result >= threshold)
│           detections.append({x, y, w, h, conf})
│       return detections
│
└── 3. nms(detections, iou_threshold=0.3)
        boxes = np.array([[x, y, x+w, y+h] for d in detections])
        scores = np.array([d['conf'] for d in detections])
        order = scores.argsort()[::-1]
        # Keep if IoU <= threshold with all previous
        return [detections[i] for i in keep]
```

---

## HITO 2: División del Dataset

**Descripción:** Dividir imágenes y labels en conjuntos train/val manteniendo pares consistentes.

**Goal:** Crear estructura de carpetas YOLO con split 80/20 reproducible.

**Inputs:**
- `data/video_frames/*.jpg` - Imágenes
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
        +-------+--------+
                |
       +--------+--------+
       |                 |
       v                 v
+------+------+   +------+------+
|   train/    |   |    val/     |
+-------------+   +-------------+
```

**Breakdown 3x3:**
```
scripts/split_dataset.py
│
├── 1. get_matching_pairs(images_dir, labels_dir)
│       pairs = []
│       for img_path in images_dir.glob('*.jpg'):
│           label_path = labels_dir / f"{img_path.stem}.txt"
│           if label_path.exists():
│               pairs.append({'image': img_path, 'label': label_path})
│       return pairs
│
├── 2. split_pairs(pairs, train_ratio=0.8, seed=42)
│       random.seed(seed)
│       shuffled = pairs.copy()
│       random.shuffle(shuffled)
│       split_idx = int(len(shuffled) * train_ratio)
│       return shuffled[:split_idx], shuffled[split_idx:]
│
└── 3. copy_files(pairs, output_dir, split_name)
        images_out = output_dir / split_name / 'images'
        labels_out = output_dir / split_name / 'labels'
        for pair in pairs:
            shutil.copy(pair['image'], images_out)
            shutil.copy(pair['label'], labels_out)
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
       +--------+----------+
                |
                v
        +-------+--------+
        | yolo_to_pixels |
        +-------+--------+
                |
                v
        +-------+---------+
        | draw_annotations|
        +-----------------+
```

**Breakdown 3x3:**
```
scripts/visualize_annotations.py
│
├── 1. load_annotations(label_path)
│       annotations = []
│       with open(label_path, 'r') as f:
│           for line in f:
│               parts = line.strip().split()
│               annotations.append({
│                   'class': int(parts[0]),
│                   'x': float(parts[1]), 'y': float(parts[2]),
│                   'w': float(parts[3]), 'h': float(parts[4])
│               })
│       return annotations
│
├── 2. yolo_to_pixels(ann, img_width, img_height)
│       x_center = ann['x'] * img_width
│       y_center = ann['y'] * img_height
│       x1 = int(x_center - ann['w'] * img_width / 2)
│       y1 = int(y_center - ann['h'] * img_height / 2)
│       return x1, y1, x2, y2
│
└── 3. draw_annotations(image, annotations, color=(0,255,0))
        for ann in annotations:
            x1, y1, x2, y2 = yolo_to_pixels(ann, w, h)
            cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
            cv2.putText(image, f"class {ann['class']}", (x1, y1-5), ...)
        return image
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
- `models/yolo12s_pillars_YYYYMMDD.pt` - Modelo fine-tuned
- `runs/train/exp/` - Logs, métricas, checkpoints

**Minimap:**
```
+-------------+     +-------------+     +-------------+
| yolo12s.pt  |     | pillar.yaml |     | config.yaml |
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
                   +-------+--------+
                           |
              +------------+------------+
              |                         |
              v                         v
      +-------+-------+         +-------+-------+
      | best.pt       |         | runs/train/   |
      +---------------+         +---------------+
```

**Breakdown 3x3:**
```
scripts/train.py
│
├── 1. check_gpu_availability() + load_config(config_path)
│       cuda_available = torch.cuda.is_available()
│       gpu_count = torch.cuda.device_count()
│       props = torch.cuda.get_device_properties(0)  # RTX 2060: 6GB
│       with open(config_path) as f:
│           config = yaml.safe_load(f)
│       return config
│
├── 2. setup_training(config)
│       from ultralytics import YOLO
│       model = YOLO(config.get('model', 'yolo12s.pt'))
│       training_params = {
│           'data': 'data/pillar.yaml',
│           'epochs': config.get('epochs', 100),
│           'batch': config.get('batch', 8),
│           'imgsz': 640, 'device': '0', 'amp': True
│       }
│
└── 3. train_and_save(model, training_params)
        results = model.train(**training_params)
        # Guardar con timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        shutil.copy(best_model_path, f"models/yolo12s_pillars_{timestamp}.pt")
```

---

## HITO 5: Inferencia

**Descripción:** Ejecutar detección en imágenes, videos o webcam usando el modelo entrenado.

**Goal:** Detectar pilares en tiempo real (~180 FPS con TensorRT).

**Inputs:**
- `models/*.pt` o `*.engine` - Modelo
- Imagen / Video / Webcam (source)

**Outputs:**
- Imagen/video con detecciones dibujadas
- Estadísticas (num detecciones, confianza, FPS)

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
| (img/vid/0) |    | conf=0.65      |
+-------------+    +-------+--------+
                           |
                           v
                   +-------+--------+
                   | results[0]     |
                   | .boxes.plot()  |
                   +-------+--------+
                           |
              +------------+------------+
              v                         v
      +-------+-------+         +-------+-------+
      | annotated.jpg |         | stats.json    |
      +---------------+         +---------------+
```

**Breakdown 3x3:**
```
scripts/inference.py
│
├── 1. find_best_model()
│       models_dir = PROJECT_ROOT / 'models'
│       # Prioridad: .engine > .pt > .onnx
│       versioned = sorted(models_dir.glob('vr_boxes_*.pt'), reverse=True)
│       if versioned: return versioned[0]
│       # Fallback: runs/train/*/weights/best.pt
│
├── 2. run_inference(args)
│       from ultralytics import YOLO
│       model = YOLO(str(model_path))
│       results = model.predict(
│           source=source,
│           conf=args.conf,      # default 0.65
│           iou=args.iou,        # default 0.45
│           save=True, stream=True
│       )
│
└── 3. process_results(results)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                logger.info(f"{model.names[cls]}: {conf:.2%}")
```

---

## HITO 6: Auto-Anotación Zero-Shot con Grounding DINO

**Descripción:** Anotar imágenes usando descripción en lenguaje natural sin entrenamiento previo.

**Goal:** Crear dataset YOLO completo desde prompt de texto ("yellow and black checkered pillar").

**Inputs:**
- `data/video_frames/*.jpg` - Imágenes sin anotar
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
          | processor()    |
          | model()        |
          +-------+--------+
                  |
                  v
          +-------+--------+
          | post_process   |
          | threshold=0.3  |
          +-------+--------+
                  |
         +--------+--------+
         |                 |
         v                 v
   +-----+-----+     +-----+-----+
   | labels/   |     | images/   |
   | *.txt     |     | train/val |
   +-----------+     +-----------+
```

**Breakdown 3x3:**
```
scripts/auto_annotate_grounding_dino.py
│
├── 1. load_grounding_dino(model_name="IDEA-Research/grounding-dino-tiny")
│       from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
│       device = "cuda" if torch.cuda.is_available() else "cpu"
│       processor = AutoProcessor.from_pretrained(model_name)
│       model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
│       model.to(device)
│
├── 2. detect_with_prompt(image, prompt, threshold=0.3)
│       prompt = prompt + "."  # Grounding DINO requiere punto final
│       inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
│       outputs = model(**inputs.to(device))
│       results = processor.post_process_grounded_object_detection(
│           outputs, input_ids=inputs.input_ids, target_sizes=[(h, w)]
│       )[0]
│       mask = results["scores"] >= threshold
│       return results["boxes"][mask], results["scores"][mask]
│
└── 3. create_yolo_dataset(source_dir, output_dir, prompt, val_split=0.2)
        # Split train/val
        random.seed(42)
        val_set = set(random.sample(images, int(len(images) * val_split)))

        for img_path in images:
            boxes, scores = detect_with_prompt(img_path, prompt)
            # Convertir a YOLO format
            yolo_line = f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

        # Generar dataset.yaml
        yaml.dump({'path': output_dir, 'train': 'images/train', ...})
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
| Unitarios   |    | Integración   |   | conftest.py |
+-------------+    +---------------+   +-------------+
                           |
                           v
               +-----------+-----------+
               |  47 tests PASSED ✓    |
               +-----------------------+
```

**Breakdown 3x3:**
```
tests/
│
├── 1. Tests Unitarios (test_*.py)
│       def test_nms_removes_overlapping():
│           detections = [{'x':100,'y':100,'w':50,'h':50,'conf':0.9}, ...]
│           result = nms(detections, iou_threshold=0.3)
│           assert len(result) == 1
│
│       def test_yolo_format_normalized():
│           lines = to_yolo_format(detections, img_w=1920, img_h=1080)
│           for line in lines:
│               values = [float(x) for x in line.split()[1:]]
│               assert all(0 <= v <= 1 for v in values)
│
├── 2. Tests de Integración (test_integration.py)
│       def test_config_exists():
│           assert (PROJECT_ROOT / 'config.yaml').exists()
│
│       def test_dataset_structure():
│           for split in ['train', 'val']:
│               assert (DATA_DIR / split / 'images').exists()
│               assert (DATA_DIR / split / 'labels').exists()
│
└── 3. Fixtures (conftest.py)
        @pytest.fixture
        def sample_image():
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        @pytest.fixture
        def sample_label_content():
            return "0 0.5 0.5 0.2 0.3\n0 0.7 0.3 0.1 0.2"
```

---

## HITO 8: GUI Robusta con Gradio

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
+-------+ +---------+ +---------+ +---------+ +---------+ +--------+ +-------+
```

**Breakdown 3x3:**
```
app.py
│
├── 1. Tab Inference (Image + Video)
│       def run_inference_image(image, model_choice, confidence, iou_threshold):
│           model = load_model(MODELS_DIR / model_choice)
│           results = model(image, conf=confidence, iou=iou_threshold)
│           annotated = results[0].plot()
│           num_detections = len(results[0].boxes)
│           return annotated, f"✅ Detections: {num_detections}"
│
│       def run_inference_video(video_path, ...):
│           cap = cv2.VideoCapture(video_path)
│           out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
│           while True:
│               results = model(frame, conf=conf)
│               out.write(results[0].plot())
│
├── 2. Tab Auto-Annotate (class AutoAnnotator)
│       class AutoAnnotator:
│           def load_model(self, model_name):
│               self.processor = AutoProcessor.from_pretrained(model_name)
│               self.model = AutoModelForZeroShotObjectDetection.from_pretrained(...)
│
│           def annotate_image(self, image, prompt, threshold):
│               inputs = self.processor(images=image, text=prompt+".")
│               outputs = self.model(**inputs)
│               # Draw boxes + return YOLO format
│
│           def annotate_folder(self, source, output, prompt, ...):
│               # Batch processing + create dataset.yaml
│
└── 3. Tab Annotations (class AnnotationReviewer)
        class AnnotationReviewer:
            def load_dataset(self, dataset_path):
                self.image_files = sorted(images_dir.glob("*.jpg"))

            def _load_current(self):
                img = cv2.imread(self.image_files[self.current_idx])
                for ann in self.annotations:
                    cv2.rectangle(img, (x1,y1), (x2,y2), GREEN, 2)

            def delete_annotation(self, ann_id):
                self.annotations = [a for a in self.annotations if a['id'] != ann_id]
                # Save updated label file
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
