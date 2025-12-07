#!/usr/bin/env python3
"""
Revisor interactivo de anotaciones.
Muestra cada imagen con sus anotaciones y permite borrar/añadir.

Controles:
  - ESPACIO: Aceptar y siguiente
  - D: Borrar todas las anotaciones de esta imagen
  - Click izquierdo en caja: Borrar esa anotación
  - A: Modo añadir (luego click + arrastrar para dibujar)
  - Q: Salir y guardar
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import shutil

class AnnotationReviewer:
    def __init__(self, images_dir, labels_dir, output_dir):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Encontrar imágenes con anotaciones
        self.image_files = sorted(self.images_dir.glob("*.jpg"))
        self.current_idx = 0
        self.annotations = []
        self.deleted_boxes = set()

        # Para dibujar nuevas cajas
        self.drawing = False
        self.add_mode = False  # Modo añadir activado con tecla A
        self.start_point = None
        self.end_point = None

    def load_annotations(self, label_path):
        """Cargar anotaciones YOLO"""
        annotations = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x, y, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        annotations.append({'id': i, 'cls': cls, 'x': x, 'y': y, 'w': w, 'h': h})
        return annotations

    def save_annotations(self, label_path, annotations):
        """Guardar anotaciones YOLO"""
        with open(label_path, 'w') as f:
            for ann in annotations:
                f.write(f"{ann['cls']} {ann['x']:.6f} {ann['y']:.6f} {ann['w']:.6f} {ann['h']:.6f}\n")

    def draw_annotations(self, img, annotations, deleted):
        """Dibujar anotaciones en la imagen"""
        h, w = img.shape[:2]
        img_copy = img.copy()

        for ann in annotations:
            if ann['id'] in deleted:
                continue

            # Convertir YOLO a píxeles
            cx, cy = int(ann['x'] * w), int(ann['y'] * h)
            bw, bh = int(ann['w'] * w), int(ann['h'] * h)
            x1, y1 = cx - bw // 2, cy - bh // 2
            x2, y2 = cx + bw // 2, cy + bh // 2

            # Dibujar
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_copy, str(ann['id']), (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Dibujar caja en progreso (amarilla)
        if self.drawing and self.start_point and self.end_point:
            cv2.rectangle(img_copy, self.start_point, self.end_point, (0, 255, 255), 2)

        return img_copy

    def add_annotation(self, x1, y1, x2, y2):
        """Añadir nueva anotación desde coordenadas de píxeles"""
        h, w = self.current_img.shape[:2]

        # Asegurar que x1 < x2 y y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Convertir a formato YOLO (normalizado, centro)
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        # Nuevo ID
        new_id = max([ann['id'] for ann in self.annotations], default=-1) + 1

        self.annotations.append({
            'id': new_id,
            'cls': 0,
            'x': cx,
            'y': cy,
            'w': bw,
            'h': bh
        })

    def get_clicked_annotation(self, x, y, img_shape, annotations, deleted):
        """Encontrar qué anotación fue clickeada"""
        h, w = img_shape[:2]

        for ann in annotations:
            if ann['id'] in deleted:
                continue

            cx, cy = int(ann['x'] * w), int(ann['y'] * h)
            bw, bh = int(ann['w'] * w), int(ann['h'] * h)
            x1, y1 = cx - bw // 2, cy - bh // 2
            x2, y2 = cx + bw // 2, cy + bh // 2

            if x1 <= x <= x2 and y1 <= y <= y2:
                return ann['id']

        return None

    def mouse_callback(self, event, x, y, flags, param):
        """Manejar clicks del ratón"""
        if self.add_mode:
            # Modo añadir - click izquierdo para dibujar
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_point = (x, y)
                self.end_point = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.end_point = (x, y)
                self.need_redraw = True

            elif event == cv2.EVENT_LBUTTONUP and self.drawing:
                self.drawing = False
                if self.start_point and self.end_point:
                    if abs(self.end_point[0] - self.start_point[0]) > 10 and \
                       abs(self.end_point[1] - self.start_point[1]) > 10:
                        self.add_annotation(self.start_point[0], self.start_point[1],
                                           self.end_point[0], self.end_point[1])
                self.start_point = None
                self.end_point = None
                self.add_mode = False  # Salir del modo añadir
                self.need_redraw = True
        else:
            # Modo normal - click izquierdo para borrar
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_id = self.get_clicked_annotation(x, y, self.current_img.shape,
                                                          self.annotations, self.deleted_boxes)
                if clicked_id is not None:
                    self.deleted_boxes.add(clicked_id)
                    self.need_redraw = True

    def run(self):
        """Ejecutar el revisor"""
        cv2.namedWindow('Revisor de Anotaciones')
        cv2.setMouseCallback('Revisor de Anotaciones', self.mouse_callback)

        stats = {'reviewed': 0, 'modified': 0, 'deleted_boxes': 0}

        while self.current_idx < len(self.image_files):
            img_path = self.image_files[self.current_idx]
            label_path = self.labels_dir / f"{img_path.stem}.txt"

            # Cargar imagen y anotaciones
            self.current_img = cv2.imread(str(img_path))
            self.annotations = self.load_annotations(label_path)
            self.deleted_boxes = set()
            self.need_redraw = True

            # Si no hay anotaciones, saltar
            if not self.annotations:
                self.current_idx += 1
                continue

            while True:
                if self.need_redraw:
                    display = self.draw_annotations(self.current_img, self.annotations, self.deleted_boxes)

                    # Info
                    info = f"[{self.current_idx + 1}/{len(self.image_files)}] {img_path.name} | "
                    info += f"Cajas: {len(self.annotations) - len(self.deleted_boxes)} | "
                    if self.add_mode:
                        info += "MODO AÑADIR: click+arrastrar"
                    else:
                        info += "ESPACIO=sig D=borrar A=añadir Q=salir"
                    cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.imshow('Revisor de Anotaciones', display)
                    self.need_redraw = False

                key = cv2.waitKey(100) & 0xFF

                if key == ord(' '):  # Espacio - aceptar y siguiente
                    break
                elif key == ord('d'):  # D - borrar todas
                    self.deleted_boxes = set(ann['id'] for ann in self.annotations)
                    self.need_redraw = True
                elif key == ord('a'):  # A - modo añadir
                    self.add_mode = True
                    self.need_redraw = True
                elif key == ord('q'):  # Q - salir
                    cv2.destroyAllWindows()
                    print(f"\nEstadísticas: {stats}")
                    return

            # Guardar anotaciones limpias
            clean_annotations = [ann for ann in self.annotations if ann['id'] not in self.deleted_boxes]
            output_label = self.output_dir / f"{img_path.stem}.txt"
            self.save_annotations(output_label, clean_annotations)

            # Copiar imagen
            shutil.copy(img_path, self.output_dir / img_path.name)

            # Stats
            stats['reviewed'] += 1
            if self.deleted_boxes:
                stats['modified'] += 1
                stats['deleted_boxes'] += len(self.deleted_boxes)

            self.current_idx += 1

        cv2.destroyAllWindows()
        print(f"\n¡Revisión completada!")
        print(f"Estadísticas: {stats}")
        print(f"Anotaciones limpias guardadas en: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Revisor interactivo de anotaciones')
    parser.add_argument('--images', required=True, help='Carpeta de imágenes')
    parser.add_argument('--labels', required=True, help='Carpeta de anotaciones')
    parser.add_argument('--output', required=True, help='Carpeta de salida (anotaciones limpias)')

    args = parser.parse_args()

    reviewer = AnnotationReviewer(args.images, args.labels, args.output)
    reviewer.run()


if __name__ == '__main__':
    main()
