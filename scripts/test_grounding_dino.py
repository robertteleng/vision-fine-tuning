#!/usr/bin/env python3
"""
Prueba de Grounding DINO usando transformers de HuggingFace.

Más simple y sin dependencias problemáticas.
"""

import argparse
from pathlib import Path
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def main():
    parser = argparse.ArgumentParser(description='Test Grounding DINO')
    parser.add_argument('--image', type=str, required=True, help='Image to test')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--threshold', type=float, default=0.25, help='Detection threshold')
    parser.add_argument('--output', type=str, default='grounding_test.jpg')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("GROUNDING DINO - Test (via HuggingFace)")
    print(f"{'='*60}")
    print(f"  Imagen: {args.image}")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Threshold: {args.threshold}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Cargar modelo desde HuggingFace
    print("\n  Descargando/cargando modelo...")
    model_id = "IDEA-Research/grounding-dino-tiny"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # Cargar imagen
    print("  Cargando imagen...")
    image = Image.open(args.image).convert("RGB")

    # Preparar inputs
    # Grounding DINO espera el prompt con punto al final
    text = args.prompt
    if not text.endswith("."):
        text = text + "."

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    # Detectar
    print("  Detectando...")
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-procesar resultados
    width, height = image.size
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        target_sizes=[(height, width)]
    )[0]

    # Filtrar por threshold
    mask = results["scores"] >= args.threshold
    boxes = results["boxes"][mask].cpu().numpy()
    scores = results["scores"][mask].cpu().numpy()
    labels = [results["labels"][i] for i, m in enumerate(mask) if m]

    # Resultados
    print(f"\n{'='*60}")
    print("RESULTADOS")
    print(f"{'='*60}")
    print(f"  Detecciones encontradas: {len(boxes)}")

    if len(boxes) > 0:
        print(f"\n  Detalles:")
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            print(f"    [{i}] '{label}' - conf: {score:.3f} - box: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

        # Dibujar en imagen
        img_cv = cv2.imread(args.image)
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
            text_label = f"{label} {score:.2f}"
            cv2.putText(img_cv, text_label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imwrite(args.output, img_cv)
        print(f"\n  Resultado guardado: {args.output}")
    else:
        print("\n  ❌ No se detectaron objetos")
        print("     Prueba con:")
        print("       - Umbral más bajo (--threshold 0.1)")
        print("       - Prompt diferente")


if __name__ == '__main__':
    main()
