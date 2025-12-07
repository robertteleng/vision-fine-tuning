#!/usr/bin/env python3
"""
VR Pillar Detector - Aplicación Gradio

Interfaz web para detección de pilares en entornos VR.

Uso:
    python app.py
    python app.py --share  # Compartir públicamente
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import gradio as gr
import numpy as np
from PIL import Image

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

APP_TITLE = "VR Pillar Detector"
APP_DESCRIPTION = """
Detección de pilares de señalización amarillo/negro en entornos de Realidad Virtual.

**Modelo:** YOLOv12s fine-tuned | **Precisión:** 98.7% mAP@50 | **Velocidad:** 180 FPS (TensorRT)
"""

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def find_available_models():
    """Encuentra todos los modelos disponibles."""
    models = []
    if MODELS_DIR.exists():
        for ext in ["*.engine", "*.pt", "*.onnx"]:
            models.extend(MODELS_DIR.glob(ext))
    return sorted(models, key=lambda x: x.suffix != ".engine")  # TensorRT primero


def get_model_info():
    """Obtiene información sobre los modelos disponibles."""
    models = find_available_models()
    if not models:
        return "No se encontraron modelos en la carpeta models/"

    info = "**Modelos disponibles:**\n\n"
    for m in models:
        size_mb = m.stat().st_size / (1024 * 1024)
        format_name = {
            ".engine": "TensorRT FP16",
            ".pt": "PyTorch",
            ".onnx": "ONNX"
        }.get(m.suffix, m.suffix)
        info += f"- `{m.name}` ({format_name}, {size_mb:.1f} MB)\n"

    return info


def load_model(model_path: str = None):
    """Carga el modelo YOLO."""
    from ultralytics import YOLO

    if model_path and Path(model_path).exists():
        return YOLO(model_path)

    # Buscar mejor modelo disponible
    models = find_available_models()
    if models:
        return YOLO(str(models[0]))

    raise FileNotFoundError("No se encontró ningún modelo")


# ============================================================================
# TAB: INFERENCIA
# ============================================================================

def run_inference_image(image, model_choice, confidence, iou_threshold):
    """Ejecuta inferencia en una imagen."""
    if image is None:
        return None, "Por favor, sube una imagen"

    try:
        # Cargar modelo
        model_path = MODELS_DIR / model_choice if model_choice else None
        model = load_model(str(model_path) if model_path else None)

        # Ejecutar inferencia
        results = model(image, conf=confidence, iou=iou_threshold, verbose=False)

        # Obtener imagen con anotaciones
        annotated = results[0].plot()

        # Estadísticas
        num_detections = len(results[0].boxes)
        if num_detections > 0:
            confs = results[0].boxes.conf.cpu().numpy()
            stats = f"**Detecciones:** {num_detections}\n\n"
            stats += f"**Confianza:** {confs.min():.2f} - {confs.max():.2f}\n\n"
            stats += f"**Media:** {confs.mean():.2f}"
        else:
            stats = "**No se detectaron pilares**"

        return annotated, stats

    except Exception as e:
        return None, f"Error: {str(e)}"


def run_inference_video(video_path, model_choice, confidence, iou_threshold, progress=gr.Progress()):
    """Ejecuta inferencia en un video."""
    import cv2
    import tempfile

    if video_path is None:
        return None, "Por favor, sube un video"

    try:
        model_path = MODELS_DIR / model_choice if model_choice else None
        model = load_model(str(model_path) if model_path else None)

        # Abrir video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Crear video de salida
        output_path = tempfile.mktemp(suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_detections = 0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inferencia
            results = model(frame, conf=confidence, iou=iou_threshold, verbose=False)
            annotated = results[0].plot()
            out.write(annotated)

            total_detections += len(results[0].boxes)
            frame_count += 1

            # Actualizar progreso
            progress(frame_count / total_frames, desc=f"Frame {frame_count}/{total_frames}")

        cap.release()
        out.release()

        stats = f"**Frames procesados:** {frame_count}\n\n"
        stats += f"**Detecciones totales:** {total_detections}\n\n"
        stats += f"**Promedio por frame:** {total_detections/frame_count:.1f}"

        return output_path, stats

    except Exception as e:
        return None, f"Error: {str(e)}"


# ============================================================================
# TAB: MÉTRICAS
# ============================================================================

def get_training_metrics():
    """Lee las métricas del último entrenamiento."""
    metrics = """
## Resultados del Modelo Final (Entrenamiento #4)

| Métrica | Valor |
|---------|-------|
| **mAP@50** | 98.7% |
| **mAP@50-95** | 87.4% |
| **Precisión** | 91.7% |
| **Recall** | 99.2% |

## Benchmark de Inferencia (RTX 2060)

| Formato | Velocidad | FPS | Speedup |
|---------|-----------|-----|---------|
| PyTorch (.pt) | 14.0 ms | 71 | 1x |
| ONNX (.onnx) | 19.4 ms | 52 | 0.7x |
| **TensorRT FP16** | **5.5 ms** | **180** | **2.5x** |

## Dataset

- **Training:** 621 imágenes
- **Validation:** 139 imágenes
- **Clases:** 1 (pillar)
"""
    return metrics


def run_benchmark(iterations, progress=gr.Progress()):
    """Ejecuta benchmark de velocidad."""
    import time
    import torch

    models = find_available_models()
    if not models:
        return "No se encontraron modelos"

    results = "## Resultados del Benchmark\n\n"
    results += f"**Iteraciones:** {iterations}\n\n"

    # Imagen dummy
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    for model_path in progress.tqdm(models, desc="Evaluando modelos"):
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))

            # Warmup
            for _ in range(5):
                model(dummy_img, verbose=False)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Benchmark
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                model(dummy_img, verbose=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            mean_ms = np.mean(times) * 1000
            fps = 1000 / mean_ms

            format_name = {
                ".engine": "TensorRT",
                ".pt": "PyTorch",
                ".onnx": "ONNX"
            }.get(model_path.suffix, model_path.suffix)

            results += f"### {format_name} (`{model_path.name}`)\n"
            results += f"- **Velocidad:** {mean_ms:.2f} ms\n"
            results += f"- **FPS:** {fps:.1f}\n\n"

        except Exception as e:
            results += f"### {model_path.name}\n- Error: {str(e)}\n\n"

    return results


# ============================================================================
# TAB: ENTRENAMIENTO
# ============================================================================

def start_training(epochs, batch_size, model_base, progress=gr.Progress()):
    """Inicia el entrenamiento del modelo."""
    try:
        from ultralytics import YOLO
        import yaml

        # Cargar configuración
        config_path = PROJECT_ROOT / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Actualizar parámetros
        config["epochs"] = epochs
        config["batch"] = batch_size

        # Cargar modelo base
        model = YOLO(model_base)

        # Dataset
        data_yaml = PROJECT_ROOT / "data" / "pillar.yaml"

        # Entrenar
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            project=str(PROJECT_ROOT / "runs" / "train"),
            verbose=True
        )

        return f"Entrenamiento completado. Resultados en: {results.save_dir}"

    except Exception as e:
        return f"Error durante el entrenamiento: {str(e)}"


def get_training_status():
    """Obtiene el estado del entrenamiento."""
    runs_dir = PROJECT_ROOT / "runs" / "train"
    if not runs_dir.exists():
        return "No hay entrenamientos previos"

    experiments = sorted(runs_dir.glob("*/"), reverse=True)
    if not experiments:
        return "No hay entrenamientos previos"

    status = "## Entrenamientos Anteriores\n\n"
    for exp in experiments[:5]:
        status += f"- `{exp.name}`\n"

        # Buscar métricas
        results_csv = exp / "results.csv"
        if results_csv.exists():
            import csv
            with open(results_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last = rows[-1]
                    status += f"  - Épocas: {len(rows)}\n"

    return status


# ============================================================================
# TAB: ANOTACIONES
# ============================================================================

class AnnotationReviewer:
    """Revisor de anotaciones para Gradio."""

    def __init__(self):
        self.images_dir = None
        self.labels_dir = None
        self.image_files = []
        self.current_idx = 0
        self.annotations = []

    def load_dataset(self, dataset_path: str):
        """Carga un dataset para revisar."""
        dataset_path = Path(dataset_path) if dataset_path else DATA_DIR / "dataset" / "train"

        self.images_dir = dataset_path / "images"
        self.labels_dir = dataset_path / "labels"

        if not self.images_dir.exists():
            return None, f"No se encontró: {self.images_dir}", "0 / 0"

        self.image_files = sorted(self.images_dir.glob("*.jpg"))
        if not self.image_files:
            self.image_files = sorted(self.images_dir.glob("*.png"))

        if not self.image_files:
            return None, "No se encontraron imágenes", "0 / 0"

        self.current_idx = 0
        return self._load_current()

    def _load_current(self):
        """Carga la imagen y anotaciones actuales."""
        if not self.image_files:
            return None, "No hay imágenes", "0 / 0"

        img_path = self.image_files[self.current_idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        # Cargar imagen
        img = cv2.imread(str(img_path))
        if img is None:
            return None, f"Error cargando: {img_path.name}", f"{self.current_idx + 1} / {len(self.image_files)}"

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Cargar anotaciones
        self.annotations = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        self.annotations.append({
                            'id': i,
                            'cls': int(parts[0]),
                            'x': float(parts[1]),
                            'y': float(parts[2]),
                            'w': float(parts[3]),
                            'h': float(parts[4])
                        })

        # Dibujar anotaciones
        img_display = img.copy()
        for ann in self.annotations:
            cx, cy = int(ann['x'] * w), int(ann['y'] * h)
            bw, bh = int(ann['w'] * w), int(ann['h'] * h)
            x1, y1 = cx - bw // 2, cy - bh // 2
            x2, y2 = cx + bw // 2, cy + bh // 2

            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_display, str(ann['id']), (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        info = f"**{img_path.name}** - {len(self.annotations)} anotaciones"
        counter = f"{self.current_idx + 1} / {len(self.image_files)}"

        return img_display, info, counter

    def next_image(self):
        """Siguiente imagen."""
        if self.image_files and self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
        return self._load_current()

    def prev_image(self):
        """Imagen anterior."""
        if self.image_files and self.current_idx > 0:
            self.current_idx -= 1
        return self._load_current()

    def goto_image(self, idx: int):
        """Ir a imagen específica."""
        if self.image_files and 0 <= idx - 1 < len(self.image_files):
            self.current_idx = idx - 1
        return self._load_current()

    def delete_annotation(self, ann_id: int):
        """Elimina una anotación y guarda."""
        if not self.image_files:
            return self._load_current()

        # Filtrar anotación
        self.annotations = [a for a in self.annotations if a['id'] != ann_id]

        # Guardar
        img_path = self.image_files[self.current_idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        with open(label_path, 'w') as f:
            for ann in self.annotations:
                f.write(f"{ann['cls']} {ann['x']:.6f} {ann['y']:.6f} {ann['w']:.6f} {ann['h']:.6f}\n")

        return self._load_current()

    def delete_all_annotations(self):
        """Elimina todas las anotaciones de la imagen actual."""
        if not self.image_files:
            return self._load_current()

        self.annotations = []

        img_path = self.image_files[self.current_idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        with open(label_path, 'w') as f:
            pass  # Archivo vacío

        return self._load_current()


# Instancia global del revisor
annotation_reviewer = AnnotationReviewer()


def load_annotations_dataset(dataset_choice):
    """Wrapper para cargar dataset."""
    if dataset_choice == "Train":
        path = DATA_DIR / "dataset" / "train"
    else:
        path = DATA_DIR / "dataset" / "val"
    return annotation_reviewer.load_dataset(str(path))


def annotations_next():
    return annotation_reviewer.next_image()


def annotations_prev():
    return annotation_reviewer.prev_image()


def annotations_goto(idx):
    return annotation_reviewer.goto_image(int(idx))


def annotations_delete(ann_id):
    return annotation_reviewer.delete_annotation(int(ann_id))


def annotations_delete_all():
    return annotation_reviewer.delete_all_annotations()


# ============================================================================
# INTERFAZ GRADIO
# ============================================================================

def create_app():
    """Crea la aplicación Gradio."""

    # Lista de modelos para dropdown
    model_choices = [m.name for m in find_available_models()]
    default_model = model_choices[0] if model_choices else None

    with gr.Blocks(title=APP_TITLE) as app:

        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)

        with gr.Tabs():
            # ----------------------------------------------------------------
            # TAB: INFERENCIA
            # ----------------------------------------------------------------
            with gr.TabItem("Inferencia", id="inference"):
                with gr.Tabs():
                    # Sub-tab: Imagen
                    with gr.TabItem("Imagen"):
                        with gr.Row():
                            with gr.Column():
                                img_input = gr.Image(
                                    label="Imagen de entrada",
                                    type="numpy"
                                )
                                with gr.Row():
                                    img_model = gr.Dropdown(
                                        choices=model_choices,
                                        value=default_model,
                                        label="Modelo"
                                    )
                                with gr.Row():
                                    img_conf = gr.Slider(
                                        0.1, 1.0, value=0.65,
                                        step=0.05,
                                        label="Confianza mínima"
                                    )
                                    img_iou = gr.Slider(
                                        0.1, 1.0, value=0.45,
                                        step=0.05,
                                        label="IoU threshold"
                                    )
                                img_btn = gr.Button("Detectar", variant="primary")

                            with gr.Column():
                                img_output = gr.Image(label="Resultado")
                                img_stats = gr.Markdown()

                        img_btn.click(
                            run_inference_image,
                            inputs=[img_input, img_model, img_conf, img_iou],
                            outputs=[img_output, img_stats]
                        )

                    # Sub-tab: Video
                    with gr.TabItem("Video"):
                        with gr.Row():
                            with gr.Column():
                                vid_input = gr.Video(label="Video de entrada")
                                with gr.Row():
                                    vid_model = gr.Dropdown(
                                        choices=model_choices,
                                        value=default_model,
                                        label="Modelo"
                                    )
                                with gr.Row():
                                    vid_conf = gr.Slider(
                                        0.1, 1.0, value=0.65,
                                        step=0.05,
                                        label="Confianza mínima"
                                    )
                                    vid_iou = gr.Slider(
                                        0.1, 1.0, value=0.45,
                                        step=0.05,
                                        label="IoU threshold"
                                    )
                                vid_btn = gr.Button("Procesar Video", variant="primary")

                            with gr.Column():
                                vid_output = gr.Video(label="Resultado")
                                vid_stats = gr.Markdown()

                        vid_btn.click(
                            run_inference_video,
                            inputs=[vid_input, vid_model, vid_conf, vid_iou],
                            outputs=[vid_output, vid_stats]
                        )

            # ----------------------------------------------------------------
            # TAB: MÉTRICAS
            # ----------------------------------------------------------------
            with gr.TabItem("Métricas", id="metrics"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Métricas del Modelo")
                        metrics_display = gr.Markdown(get_training_metrics())

                    with gr.Column():
                        gr.Markdown("## Modelos Disponibles")
                        models_info = gr.Markdown(get_model_info())

                gr.Markdown("---")
                gr.Markdown("## Benchmark en Vivo")

                with gr.Row():
                    bench_iterations = gr.Slider(
                        10, 200, value=50,
                        step=10,
                        label="Iteraciones"
                    )
                    bench_btn = gr.Button("Ejecutar Benchmark", variant="secondary")

                bench_results = gr.Markdown()

                bench_btn.click(
                    run_benchmark,
                    inputs=[bench_iterations],
                    outputs=[bench_results]
                )

            # ----------------------------------------------------------------
            # TAB: ENTRENAMIENTO
            # ----------------------------------------------------------------
            with gr.TabItem("Entrenamiento", id="training"):
                gr.Markdown("## Configuración de Entrenamiento")

                with gr.Row():
                    with gr.Column():
                        train_epochs = gr.Slider(
                            10, 200, value=50,
                            step=10,
                            label="Épocas"
                        )
                        train_batch = gr.Slider(
                            4, 16, value=8,
                            step=2,
                            label="Batch Size"
                        )
                        train_model = gr.Dropdown(
                            choices=["yolo12s.pt", "yolo12n.pt", "yolo11s.pt", "yolo11n.pt"],
                            value="yolo12s.pt",
                            label="Modelo Base"
                        )
                        train_btn = gr.Button("Iniciar Entrenamiento", variant="primary")

                    with gr.Column():
                        gr.Markdown("## Estado")
                        train_status = gr.Markdown(get_training_status())

                train_output = gr.Markdown()

                train_btn.click(
                    start_training,
                    inputs=[train_epochs, train_batch, train_model],
                    outputs=[train_output]
                )

            # ----------------------------------------------------------------
            # TAB: ANOTACIONES
            # ----------------------------------------------------------------
            with gr.TabItem("Anotaciones", id="annotations"):
                gr.Markdown("## Revisor de Anotaciones")
                gr.Markdown("Revisa y edita las anotaciones del dataset.")

                with gr.Row():
                    ann_dataset = gr.Radio(
                        choices=["Train", "Val"],
                        value="Train",
                        label="Dataset"
                    )
                    ann_load_btn = gr.Button("Cargar Dataset", variant="primary")

                with gr.Row():
                    with gr.Column(scale=3):
                        ann_image = gr.Image(label="Imagen con anotaciones")

                    with gr.Column(scale=1):
                        ann_info = gr.Markdown("Selecciona un dataset")
                        ann_counter = gr.Textbox(label="Progreso", value="0 / 0", interactive=False)

                        with gr.Row():
                            ann_prev_btn = gr.Button("< Anterior")
                            ann_next_btn = gr.Button("Siguiente >")

                        ann_goto = gr.Number(label="Ir a imagen #", value=1, precision=0)
                        ann_goto_btn = gr.Button("Ir")

                        gr.Markdown("---")
                        gr.Markdown("### Eliminar anotaciones")
                        ann_delete_id = gr.Number(label="ID de anotación", value=0, precision=0)
                        ann_delete_btn = gr.Button("Eliminar anotacion", variant="secondary")
                        ann_delete_all_btn = gr.Button("Eliminar TODAS", variant="stop")

                # Event handlers
                ann_load_btn.click(
                    load_annotations_dataset,
                    inputs=[ann_dataset],
                    outputs=[ann_image, ann_info, ann_counter]
                )
                ann_next_btn.click(
                    annotations_next,
                    outputs=[ann_image, ann_info, ann_counter]
                )
                ann_prev_btn.click(
                    annotations_prev,
                    outputs=[ann_image, ann_info, ann_counter]
                )
                ann_goto_btn.click(
                    annotations_goto,
                    inputs=[ann_goto],
                    outputs=[ann_image, ann_info, ann_counter]
                )
                ann_delete_btn.click(
                    annotations_delete,
                    inputs=[ann_delete_id],
                    outputs=[ann_image, ann_info, ann_counter]
                )
                ann_delete_all_btn.click(
                    annotations_delete_all,
                    outputs=[ann_image, ann_info, ann_counter]
                )

            # ----------------------------------------------------------------
            # TAB: INFORMACIÓN
            # ----------------------------------------------------------------
            with gr.TabItem("Info", id="info"):
                gr.Markdown("""
## Acerca del Proyecto

**VR Pillar Detector** es un modelo de detección de objetos especializado en identificar
pilares de señalización amarillo/negro en entornos de Realidad Virtual.

### Características

- **Modelo:** YOLOv12s fine-tuned
- **Dataset:** 760 imágenes anotadas manualmente
- **Precisión:** 98.7% mAP@50
- **Velocidad:** 180 FPS con TensorRT FP16

### Hardware Recomendado

- GPU NVIDIA con CUDA (RTX 2060 o superior)
- 6GB+ VRAM para entrenamiento
- 2GB+ VRAM para inferencia

### Uso por CLI

```bash
# Inferencia en imagen
python scripts/inference.py --source imagen.jpg

# Inferencia en video
python scripts/inference.py --source video.mp4 --model models/best.engine

# Entrenamiento
python scripts/train.py

# Benchmark
python scripts/benchmark.py
```

### Enlaces

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [YOLOv12](https://docs.ultralytics.com/models/yolo12/)
                """)

        gr.Markdown("---")
        gr.Markdown("*Desarrollado con Ultralytics YOLO y Gradio*")

    return app


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="VR Pillar Detector - Interfaz Web")
    parser.add_argument("--share", action="store_true", help="Crear enlace público")
    parser.add_argument("--port", type=int, default=7860, help="Puerto (default: 7860)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
