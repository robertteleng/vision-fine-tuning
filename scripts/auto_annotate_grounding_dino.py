#!/usr/bin/env python3
"""
Auto-anotación Zero-Shot usando Grounding DINO.

Grounding DINO funciona MEJOR que YOLO-World para objetos específicos
porque usa un modelo de lenguaje más potente para entender los prompts.

Ejemplo de uso:
    python scripts/auto_annotate_grounding_dino.py \
        --source data/video_frames/ \
        --prompt "yellow and black checkered box" \
        --output data/dataset_gdino/ \
        --threshold 0.3
"""

import argparse
from pathlib import Path
from datetime import datetime
import shutil
import random
import yaml

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def parse_args():
    parser = argparse.ArgumentParser(
        description='Auto-anotación Zero-Shot con Grounding DINO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--source', type=str, required=True,
                        help='Carpeta con imágenes')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Descripción del objeto a detectar')
    parser.add_argument('--output', type=str, required=True,
                        help='Carpeta de salida para dataset')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Umbral de confianza (default: 0.3)')
    parser.add_argument('--class-name', type=str, default=None,
                        help='Nombre de clase para labels (default: usa prompt)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fracción para validación (default: 0.2)')
    parser.add_argument('--visualize', type=int, default=10,
                        help='Guardar N imágenes visualizadas (default: 10)')
    parser.add_argument('--model', type=str, default='IDEA-Research/grounding-dino-tiny',
                        help='Modelo a usar (tiny, base)')
    parser.add_argument('--copy-images', action='store_true',
                        help='Copiar imágenes (default: symlinks)')

    return parser.parse_args()


def setup_dirs(output_path: Path) -> dict:
    """Crear estructura de carpetas YOLO."""
    dirs = {
        'images_train': output_path / 'images' / 'train',
        'images_val': output_path / 'images' / 'val',
        'labels_train': output_path / 'labels' / 'train',
        'labels_val': output_path / 'labels' / 'val',
        'viz': output_path / 'visualizations',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def find_images(source: Path) -> list:
    """Encontrar imágenes."""
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in exts:
        images.extend(source.glob(ext))
        images.extend(source.glob(ext.upper()))
    return sorted(images)


def box_to_yolo(box, img_width: int, img_height: int) -> str:
    """Convertir box [x1,y1,x2,y2] a formato YOLO normalizado."""
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    # Clase 0 (solo una clase)
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def main():
    args = parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)

    print(f"\n{'#'*60}")
    print("#  AUTO-ANOTACIÓN CON GROUNDING DINO")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    print(f"\n📂 Fuente:    {source_path}")
    print(f"📂 Salida:    {output_path}")
    print(f"📝 Prompt:    '{args.prompt}'")
    print(f"🎯 Threshold: {args.threshold}")

    # Setup
    dirs = setup_dirs(output_path)
    images = find_images(source_path)
    print(f"\n🖼️  Imágenes encontradas: {len(images)}")

    if len(images) == 0:
        print("❌ No se encontraron imágenes")
        return

    # Cargar modelo
    print(f"\n{'='*60}")
    print("Cargando Grounding DINO...")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Modelo: {args.model}")

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device)

    # Preparar prompt (Grounding DINO necesita punto al final)
    prompt = args.prompt if args.prompt.endswith(".") else args.prompt + "."

    # Split train/val
    random.seed(42)
    images_shuffled = images.copy()
    random.shuffle(images_shuffled)
    val_count = int(len(images) * args.val_split)
    val_set = set(images_shuffled[:val_count])

    # Estadísticas
    stats = {
        'total': len(images),
        'with_detections': 0,
        'total_detections': 0,
        'train': 0,
        'val': 0,
    }

    viz_count = 0

    # Procesar
    print(f"\n{'='*60}")
    print("Procesando imágenes...")
    print(f"{'='*60}")

    for img_path in tqdm(images, desc="Anotando"):
        # Cargar imagen
        pil_image = Image.open(img_path).convert("RGB")
        width, height = pil_image.size

        # Inferencia
        inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            target_sizes=[(height, width)]
        )[0]

        # Filtrar por threshold
        mask = results["scores"] >= args.threshold
        boxes = results["boxes"][mask].cpu().numpy()
        scores = results["scores"][mask].cpu().numpy()

        # Determinar split
        is_val = img_path in val_set
        split = 'val' if is_val else 'train'

        if is_val:
            stats['val'] += 1
        else:
            stats['train'] += 1

        # Guardar labels
        label_dir = dirs['labels_val'] if is_val else dirs['labels_train']
        label_path = label_dir / f"{img_path.stem}.txt"

        yolo_lines = [box_to_yolo(box, width, height) for box in boxes]

        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        # Actualizar stats
        if len(boxes) > 0:
            stats['with_detections'] += 1
        stats['total_detections'] += len(boxes)

        # Copiar/enlazar imagen
        img_dir = dirs['images_val'] if is_val else dirs['images_train']
        img_dest = img_dir / img_path.name

        if args.copy_images:
            shutil.copy2(img_path, img_dest)
        else:
            if not img_dest.exists():
                img_dest.symlink_to(img_path.absolute())

        # Visualización
        if args.visualize > 0 and viz_count < args.visualize and len(boxes) > 0:
            img_cv = cv2.imread(str(img_path))
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img_cv, f"{score:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            viz_path = dirs['viz'] / f"viz_{img_path.stem}.jpg"
            cv2.imwrite(str(viz_path), img_cv)
            viz_count += 1

    # Crear dataset.yaml
    class_name = args.class_name if args.class_name else args.prompt.replace(" ", "_").replace(".", "").lower()

    yaml_content = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: class_name},
    }

    yaml_path = output_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    # Resumen
    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    print(f"\n📊 Estadísticas:")
    print(f"  Total imágenes:           {stats['total']}")
    print(f"  Con detecciones:          {stats['with_detections']}")
    print(f"  Sin detecciones:          {stats['total'] - stats['with_detections']}")
    print(f"  Total detecciones:        {stats['total_detections']}")

    if stats['total'] > 0:
        coverage = stats['with_detections'] / stats['total'] * 100
        avg = stats['total_detections'] / stats['total']
        print(f"  Cobertura:                {coverage:.1f}%")
        print(f"  Promedio por imagen:      {avg:.2f}")

    print(f"\n📁 Split:")
    print(f"  Train: {stats['train']}")
    print(f"  Val:   {stats['val']}")

    print(f"\n📄 Dataset YAML: {yaml_path}")

    print(f"\n{'='*60}")
    print("SIGUIENTE PASO")
    print(f"{'='*60}")
    print(f"""
Entrenar con:

    python scripts/train.py \\
        --data {yaml_path} \\
        --model yolo12n.pt \\
        --epochs 100 \\
        --batch 16
""")


if __name__ == '__main__':
    main()
