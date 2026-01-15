import cv2
import numpy as np
import shutil
import random
import yaml
from pathlib import Path
from PIL import Image as PILImage
try:
    import gradio as gr
except ImportError:
    gr = None

class AutoAnnotator:
    """Auto-annotation using Grounding DINO."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None

    def load_model(self, model_name: str = "IDEA-Research/grounding-dino-tiny"):
        """Load Grounding DINO model."""
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device)
        return f"Model loaded on {self.device}"

    def annotate_image(self, image, prompt: str, threshold: float = 0.3):
        """Annotate a single image."""
        if image is None:
            return None, "No image provided", ""

        if self.model is None:
            return None, "Model not loaded. Click 'Load Model' first.", ""

        import torch

        # Ensure prompt ends with period
        prompt = prompt if prompt.endswith(".") else prompt + "."

        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image

        width, height = pil_image.size

        # Inference
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            target_sizes=[(height, width)]
        )[0]

        # Filter by threshold
        mask = results["scores"] >= threshold
        boxes = results["boxes"][mask].cpu().numpy()
        scores = results["scores"][mask].cpu().numpy()

        # Draw on image
        img_display = np.array(pil_image).copy()
        yolo_lines = []

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img_display, f"{score:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # YOLO format
            x_center = ((box[0] + box[2]) / 2) / width
            y_center = ((box[1] + box[3]) / 2) / height
            w = (box[2] - box[0]) / width
            h = (box[3] - box[1]) / height
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        stats = f"**Detections:** {len(boxes)}"
        if len(boxes) > 0:
            stats += f"\n**Confidence range:** {scores.min():.2f} - {scores.max():.2f}"

        yolo_output = "\n".join(yolo_lines) if yolo_lines else "# No detections"

        return img_display, stats, yolo_output

    def annotate_folder(self, source_folder: str, output_folder: str, prompt: str,
                        threshold: float, val_split: float, copy_images: bool,
                        progress=gr.Progress() if gr else None):
        """Annotate entire folder and create YOLO dataset."""
        import torch

        if self.model is None:
            return "Error: Model not loaded. Click 'Load Model' first."

        source_path = Path(source_folder)
        output_path = Path(output_folder)

        if not source_path.exists():
            return f"Error: Source folder not found: {source_folder}"

        # Find images
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in exts:
            images.extend(source_path.glob(ext))
            images.extend(source_path.glob(ext.upper()))
        images = sorted(images)

        if len(images) == 0:
            return f"Error: No images found in {source_folder}"

        # Setup directories
        dirs = {
            'images_train': output_path / 'images' / 'train',
            'images_val': output_path / 'images' / 'val',
            'labels_train': output_path / 'labels' / 'train',
            'labels_val': output_path / 'labels' / 'val',
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # Prepare prompt
        prompt = prompt if prompt.endswith(".") else prompt + "."

        # Split train/val
        random.seed(42)
        images_shuffled = images.copy()
        random.shuffle(images_shuffled)
        val_count = int(len(images) * val_split)
        val_set = set(images_shuffled[:val_count])

        # Stats
        stats = {'total': len(images), 'with_detections': 0, 'total_detections': 0, 'train': 0, 'val': 0}

        # Process
        iterator = images
        if progress:
            iterator = progress.tqdm(images, desc="Annotating")
            
        for i, img_path in enumerate(iterator):
            pil_image = PILImage.open(img_path).convert("RGB")
            width, height = pil_image.size

            # Inference
            inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs, input_ids=inputs.input_ids, target_sizes=[(height, width)]
            )[0]

            mask = results["scores"] >= threshold
            boxes = results["boxes"][mask].cpu().numpy()

            # Determine split
            is_val = img_path in val_set
            split = 'val' if is_val else 'train'
            stats[split] += 1

            # Save labels
            label_dir = dirs['labels_val'] if is_val else dirs['labels_train']
            label_path = label_dir / f"{img_path.stem}.txt"

            yolo_lines = []
            for box in boxes:
                x_center = ((box[0] + box[2]) / 2) / width
                y_center = ((box[1] + box[3]) / 2) / height
                w = (box[2] - box[0]) / width
                h = (box[3] - box[1]) / height
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

            if len(boxes) > 0:
                stats['with_detections'] += 1
            stats['total_detections'] += len(boxes)

            # Copy/link image
            img_dir = dirs['images_val'] if is_val else dirs['images_train']
            img_dest = img_dir / img_path.name

            if copy_images:
                shutil.copy2(img_path, img_dest)
            else:
                if not img_dest.exists():
                    try:
                        img_dest.symlink_to(img_path.absolute())
                    except OSError:
                        shutil.copy2(img_path, img_dest)

        # Create dataset.yaml
        yaml_content = {
            'path': str(output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'pillar'},
        }
        yaml_path = output_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        # Return summary
        coverage = stats['with_detections'] / stats['total'] * 100 if stats['total'] > 0 else 0
        return f"""## Annotation Complete!

**Statistics:**
- Total images: {stats['total']}
- With detections: {stats['with_detections']} ({coverage:.1f}%)
- Total detections: {stats['total_detections']}
- Train: {stats['train']} | Val: {stats['val']}

**Dataset YAML:** `{yaml_path}`

**Next step:** Train with:
```
python scripts/train.py --data {yaml_path}
```
"""


class AnnotationReviewer:
    """Annotation reviewer for Gradio."""

    def __init__(self):
        self.images_dir = None
        self.labels_dir = None
        self.image_files = []
        self.current_idx = 0
        self.annotations = []

    def load_dataset(self, dataset_path: str):
        """Load a dataset for review."""
        # dataset_path is expected to be a string or Path object pointing to the train or val folder (e.g. data/dataset/train)
        if not dataset_path:
             return None, "No path provided", "0 / 0"
             
        dataset_path = Path(dataset_path) 
        self.images_dir = dataset_path / "images"
        self.labels_dir = dataset_path / "labels"

        if not self.images_dir.exists():
            return None, f"Not found: {self.images_dir}", "0 / 0"

        self.image_files = sorted(self.images_dir.glob("*.jpg"))
        if not self.image_files:
            self.image_files = sorted(self.images_dir.glob("*.png"))

        if not self.image_files:
            return None, "No images found", "0 / 0"

        self.current_idx = 0
        return self._load_current()

    def _load_current(self):
        """Load current image and annotations."""
        if not self.image_files:
            return None, "No images", "0 / 0"

        img_path = self.image_files[self.current_idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            return None, f"Error loading: {img_path.name}", f"{self.current_idx + 1} / {len(self.image_files)}"

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Load annotations
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

        # Draw annotations
        img_display = img.copy()
        for ann in self.annotations:
            cx, cy = int(ann['x'] * w), int(ann['y'] * h)
            bw, bh = int(ann['w'] * w), int(ann['h'] * h)
            x1, y1 = cx - bw // 2, cy - bh // 2
            x2, y2 = cx + bw // 2, cy + bh // 2

            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_display, str(ann['id']), (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        info = f"**{img_path.name}** - {len(self.annotations)} annotations"
        counter = f"{self.current_idx + 1} / {len(self.image_files)}"

        return img_display, info, counter

    def next_image(self):
        """Next image."""
        if self.image_files and self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
        return self._load_current()

    def prev_image(self):
        """Previous image."""
        if self.image_files and self.current_idx > 0:
            self.current_idx -= 1
        return self._load_current()

    def goto_image(self, idx: int):
        """Go to specific image."""
        if self.image_files and 0 <= idx - 1 < len(self.image_files):
            self.current_idx = idx - 1
        return self._load_current()

    def delete_annotation(self, ann_id: int):
        """Delete an annotation and save."""
        if not self.image_files:
            return self._load_current()

        # Filter annotation
        self.annotations = [a for a in self.annotations if a['id'] != ann_id]

        # Save
        img_path = self.image_files[self.current_idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        with open(label_path, 'w') as f:
            for ann in self.annotations:
                f.write(f"{ann['cls']} {ann['x']:.6f} {ann['y']:.6f} {ann['w']:.6f} {ann['h']:.6f}\n")

        return self._load_current()

    def delete_all_annotations(self):
        """Delete all annotations from current image."""
        if not self.image_files:
            return self._load_current()

        self.annotations = []

        img_path = self.image_files[self.current_idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        with open(label_path, 'w') as f:
            pass  # Empty file

        return self._load_current()
