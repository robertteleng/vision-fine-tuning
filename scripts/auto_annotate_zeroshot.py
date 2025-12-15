#!/usr/bin/env python3
"""
Auto-anotación Zero-Shot usando YOLO-World.

Este script usa YOLO-World para detectar objetos basándose en descripciones
de texto (prompts) en lugar de necesitar ejemplos etiquetados.

CÓMO FUNCIONA:
1. Cargas YOLO-World (modelo pre-entrenado con CLIP)
2. Le dices QUÉ buscar con texto: "yellow checkered box"
3. YOLO-World detecta esos objetos en tus imágenes
4. Genera archivos .txt en formato YOLO para entrenar

Ejemplo de uso:
    python scripts/auto_annotate_zeroshot.py \
        --source data/video_frames/ \
        --prompts "yellow checkered pillar" \
        --output data/dataset_zeroshot/ \
        --conf 0.3
"""

import argparse
from pathlib import Path
from datetime import datetime
import shutil
import yaml

# ============================================================================
# PASO 1: Importar YOLO-World desde Ultralytics
# ============================================================================
# Ultralytics incluye YOLO-World, que combina YOLO con CLIP para
# entender descripciones de texto y buscar objetos que coincidan.
from ultralytics import YOLO


def parse_args():
    """Parsear argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Auto-anotación Zero-Shot con YOLO-World',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Detectar un tipo de objeto:
  python scripts/auto_annotate_zeroshot.py \\
      --source data/video_frames/ \\
      --prompts "yellow checkered pillar" \\
      --output data/dataset_zeroshot/

  # Detectar múltiples objetos:
  python scripts/auto_annotate_zeroshot.py \\
      --source data/video_frames/ \\
      --prompts "yellow pillar" "blue box" "orange cone" \\
      --output data/dataset_zeroshot/ \\
      --conf 0.25

  # Usar modelo más grande para mejor precisión:
  python scripts/auto_annotate_zeroshot.py \\
      --source data/video_frames/ \\
      --prompts "yellow checkered pillar" \\
      --model yolov8l-worldv2.pt \\
      --conf 0.2
        """
    )

    # Argumentos requeridos
    parser.add_argument(
        '--source', type=str, required=True,
        help='Carpeta con imágenes a anotar (jpg, png)'
    )
    parser.add_argument(
        '--prompts', type=str, nargs='+', required=True,
        help='Descripción(es) de los objetos a detectar. Cada prompt es una clase.'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Carpeta de salida para el dataset'
    )

    # Argumentos opcionales
    parser.add_argument(
        '--model', type=str, default='yolov8x-worldv2.pt',
        help='Modelo YOLO-World a usar (default: yolov8x-worldv2.pt)'
    )
    parser.add_argument(
        '--conf', type=float, default=0.3,
        help='Umbral de confianza mínimo (0-1). Menor=más detecciones. Default: 0.3'
    )
    parser.add_argument(
        '--iou', type=float, default=0.5,
        help='Umbral IoU para NMS. Default: 0.5'
    )
    parser.add_argument(
        '--imgsz', type=int, default=640,
        help='Tamaño de imagen para inferencia. Default: 640'
    )
    parser.add_argument(
        '--device', type=str, default='0',
        help='Device: "0" para GPU, "cpu" para CPU. Default: 0'
    )
    parser.add_argument(
        '--class-names', type=str, nargs='+', default=None,
        help='Nombres de clase para el dataset (si no se especifica, usa los prompts)'
    )
    parser.add_argument(
        '--val-split', type=float, default=0.2,
        help='Fracción para validación (0-1). Default: 0.2'
    )
    parser.add_argument(
        '--copy-images', action='store_true',
        help='Copiar imágenes al dataset (si no, crea symlinks)'
    )
    parser.add_argument(
        '--visualize', type=int, default=0,
        help='Guardar N imágenes con detecciones visualizadas. Default: 0'
    )

    return parser.parse_args()


def setup_output_dirs(output_path: Path) -> dict:
    """
    Crear estructura de carpetas para dataset YOLO.

    Estructura creada:
        output/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── dataset.yaml
    """
    dirs = {
        'images_train': output_path / 'images' / 'train',
        'images_val': output_path / 'images' / 'val',
        'labels_train': output_path / 'labels' / 'train',
        'labels_val': output_path / 'labels' / 'val',
        'visualize': output_path / 'visualizations',
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def load_yolo_world(model_name: str, prompts: list, device: str) -> YOLO:
    """
    Cargar YOLO-World y configurar las clases a detectar.

    EXPLICACIÓN:
    - YOLO-World usa CLIP internamente para entender texto
    - Cuando hacemos model.set_classes(prompts), le decimos qué buscar
    - El modelo convierte cada prompt a un "embedding" (vector numérico)
    - Luego compara ese vector con lo que ve en la imagen
    """
    print(f"\n{'='*60}")
    print("PASO 1: Cargando modelo YOLO-World")
    print(f"{'='*60}")
    print(f"  Modelo: {model_name}")
    print(f"  Device: {device}")

    # Cargar el modelo
    model = YOLO(model_name)

    # ¡CLAVE! Configurar qué objetos queremos detectar
    # Esto es lo que hace "zero-shot": no necesitamos entrenar,
    # solo decirle qué buscar con texto
    print(f"\n  Configurando clases a detectar:")
    for i, prompt in enumerate(prompts):
        print(f"    Clase {i}: '{prompt}'")

    model.set_classes(prompts)

    print(f"\n  ✓ Modelo cargado y configurado")
    return model


def find_images(source_path: Path) -> list:
    """Encontrar todas las imágenes en el directorio."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    images = []
    for ext in extensions:
        images.extend(source_path.glob(ext))
        images.extend(source_path.glob(ext.upper()))
    return sorted(images)


def detection_to_yolo_format(box, img_width: int, img_height: int) -> str:
    """
    Convertir una detección al formato YOLO.

    Formato YOLO: class_id x_center y_center width height
    - Todos los valores están normalizados (0-1)
    - x_center, y_center: centro del bounding box
    - width, height: dimensiones del box
    """
    # Extraer datos de la detección
    class_id = int(box.cls[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    # Calcular centro y dimensiones normalizadas
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def process_images(
    model: YOLO,
    images: list,
    dirs: dict,
    args,
) -> dict:
    """
    Procesar todas las imágenes y generar anotaciones.

    Para cada imagen:
    1. YOLO-World busca objetos que coincidan con los prompts
    2. Filtra detecciones por confianza
    3. Guarda el archivo .txt con las anotaciones
    4. Copia/enlaza la imagen al dataset
    """
    from tqdm import tqdm
    import random

    print(f"\n{'='*60}")
    print("PASO 2: Procesando imágenes")
    print(f"{'='*60}")
    print(f"  Total imágenes: {len(images)}")
    print(f"  Umbral confianza: {args.conf}")
    print(f"  Tamaño inferencia: {args.imgsz}")

    # Estadísticas
    stats = {
        'total_images': len(images),
        'images_with_detections': 0,
        'total_detections': 0,
        'detections_per_class': {},
        'train_images': 0,
        'val_images': 0,
    }

    # Mezclar imágenes para split aleatorio
    images_shuffled = images.copy()
    random.seed(42)  # Para reproducibilidad
    random.shuffle(images_shuffled)

    # Calcular índice de corte para train/val
    val_count = int(len(images_shuffled) * args.val_split)
    val_images = set(images_shuffled[:val_count])

    # Contador para visualizaciones
    viz_count = 0

    # Procesar cada imagen
    for img_path in tqdm(images, desc="Anotando"):
        # Determinar si es train o val
        is_val = img_path in val_images
        split = 'val' if is_val else 'train'

        # ============================================================
        # AQUÍ ESTÁ LA MAGIA: YOLO-World hace la detección
        # ============================================================
        # model.predict() busca objetos que coincidan con los prompts
        # que configuramos con set_classes()
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

        # Obtener el resultado (solo hay uno porque procesamos una imagen)
        result = results[0]
        boxes = result.boxes

        # Obtener dimensiones de la imagen
        img_height, img_width = result.orig_shape

        # Convertir detecciones a formato YOLO
        yolo_lines = []
        for box in boxes:
            line = detection_to_yolo_format(box, img_width, img_height)
            yolo_lines.append(line)

            # Actualizar estadísticas
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            stats['detections_per_class'][class_name] = \
                stats['detections_per_class'].get(class_name, 0) + 1

        # Actualizar estadísticas
        if len(yolo_lines) > 0:
            stats['images_with_detections'] += 1
        stats['total_detections'] += len(yolo_lines)

        if is_val:
            stats['val_images'] += 1
        else:
            stats['train_images'] += 1

        # Guardar archivo de etiquetas (.txt)
        label_dir = dirs['labels_val'] if is_val else dirs['labels_train']
        label_path = label_dir / f"{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        # Copiar o enlazar imagen
        img_dir = dirs['images_val'] if is_val else dirs['images_train']
        img_dest = img_dir / img_path.name

        if args.copy_images:
            shutil.copy2(img_path, img_dest)
        else:
            # Crear symlink (más rápido y ahorra espacio)
            if not img_dest.exists():
                img_dest.symlink_to(img_path.absolute())

        # Guardar visualización si se solicitó
        if args.visualize > 0 and viz_count < args.visualize and len(yolo_lines) > 0:
            viz_path = dirs['visualize'] / f"viz_{img_path.stem}.jpg"
            result.save(str(viz_path))
            viz_count += 1

    return stats


def create_dataset_yaml(output_path: Path, class_names: list) -> Path:
    """
    Crear archivo dataset.yaml para entrenamiento YOLO.

    Este archivo le dice a YOLO:
    - Dónde están las imágenes
    - Cuántas clases hay
    - Cómo se llama cada clase
    """
    yaml_content = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)},
    }

    yaml_path = output_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    return yaml_path


def print_summary(stats: dict, yaml_path: Path, class_names: list):
    """Imprimir resumen de la anotación."""
    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")

    print(f"\n📊 Estadísticas:")
    print(f"  Total imágenes procesadas: {stats['total_images']}")
    print(f"  Imágenes con detecciones:  {stats['images_with_detections']}")
    print(f"  Imágenes sin detecciones:  {stats['total_images'] - stats['images_with_detections']}")
    print(f"  Total detecciones:         {stats['total_detections']}")

    if stats['total_images'] > 0:
        avg = stats['total_detections'] / stats['total_images']
        coverage = stats['images_with_detections'] / stats['total_images'] * 100
        print(f"  Promedio por imagen:       {avg:.2f}")
        print(f"  Cobertura:                 {coverage:.1f}%")

    print(f"\n📁 Split del dataset:")
    print(f"  Train: {stats['train_images']} imágenes")
    print(f"  Val:   {stats['val_images']} imágenes")

    print(f"\n🏷️ Detecciones por clase:")
    for class_name, count in stats['detections_per_class'].items():
        print(f"  {class_name}: {count}")

    print(f"\n📄 Dataset YAML: {yaml_path}")

    print(f"\n{'='*60}")
    print("SIGUIENTE PASO: Entrenar tu modelo")
    print(f"{'='*60}")
    print(f"""
Para entrenar con estas anotaciones:

    python scripts/train.py \\
        --data {yaml_path} \\
        --model yolo12n.pt \\
        --epochs 100 \\
        --batch 16

O directamente con Ultralytics:

    from ultralytics import YOLO
    model = YOLO('yolo12n.pt')
    model.train(data='{yaml_path}', epochs=100, batch=16)
""")


def main():
    """Función principal."""
    args = parse_args()

    # Convertir paths
    source_path = Path(args.source)
    output_path = Path(args.output)

    # Validar que existe el directorio fuente
    if not source_path.exists():
        raise FileNotFoundError(f"No existe el directorio: {source_path}")

    print(f"\n{'#'*60}")
    print("#  AUTO-ANOTACIÓN ZERO-SHOT CON YOLO-WORLD")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    print(f"\n📂 Fuente:  {source_path}")
    print(f"📂 Salida:  {output_path}")
    print(f"📝 Prompts: {args.prompts}")

    # Crear estructura de carpetas
    dirs = setup_output_dirs(output_path)

    # Encontrar imágenes
    images = find_images(source_path)
    if len(images) == 0:
        raise ValueError(f"No se encontraron imágenes en {source_path}")
    print(f"\n🖼️ Imágenes encontradas: {len(images)}")

    # Cargar modelo
    model = load_yolo_world(args.model, args.prompts, args.device)

    # Procesar imágenes
    stats = process_images(model, images, dirs, args)

    # Crear dataset.yaml
    class_names = args.class_names if args.class_names else args.prompts
    # Limpiar nombres de clase (sin espacios, lowercase)
    class_names = [name.replace(' ', '_').lower() for name in class_names]
    yaml_path = create_dataset_yaml(output_path, class_names)

    # Mostrar resumen
    print_summary(stats, yaml_path, class_names)


if __name__ == '__main__':
    main()
