from pathlib import Path
import numpy as np
try:
    import gradio as gr
except ImportError:
    gr = None

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

def find_available_models():
    """Find all available models."""
    models = []
    if MODELS_DIR.exists():
        for ext in ["*.engine", "*.pt", "*.onnx"]:
            models.extend(MODELS_DIR.glob(ext))
    return sorted(models, key=lambda x: x.suffix != ".engine")  # TensorRT first


def run_benchmark(iterations, progress=gr.Progress() if gr else None):
    """Run speed benchmark."""
    import time
    import torch

    models = find_available_models()
    if not models:
        return "No models found"

    results = "## Benchmark Results\n\n"
    results += f"**Iterations:** {iterations}\n\n"

    # Dummy image
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    iterator = models
    if progress:
        iterator = progress.tqdm(models, desc="Evaluating models")

    for model_path in iterator:
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
            results += f"- **Speed:** {mean_ms:.2f} ms\n"
            results += f"- **FPS:** {fps:.1f}\n\n"

        except Exception as e:
            results += f"### {model_path.name}\n- Error: {str(e)}\n\n"

    return results
