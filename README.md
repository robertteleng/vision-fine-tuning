# Fine-Tuning YOLO12 para Detección de Pilares VR

## Objetivo del Proyecto

Modelo de detección de objetos especializado en identificar **pilares de señalización amarillo/negro** en entornos de Realidad Virtual. El modelo está optimizado para tiempo real usando YOLOv12s.

## Resultados Actuales

| Métrica | Valor |
|---------|-------|
| mAP@50 | 98.7% |
| mAP@50-95 | 87.4% |
| Precisión | 91.7% |
| Recall | 99.2% |
| Inferencia PyTorch | 14ms/imagen (71 FPS) |
| **Inferencia TensorRT FP16** | **5.5ms/imagen (180 FPS)** |

---

## Hardware

| Componente | Especificación |
|------------|----------------|
| GPU | NVIDIA RTX 2060 (6GB VRAM) |
| CUDA | 13.0 |
| Driver | 580.95.05 |

---

## Estructura del Proyecto

```
fine-tuning-vr/
├── config.yaml                  # Configuración de entrenamiento
├── data/
│   ├── dataset/                 # Dataset principal
│   │   ├── train/images/        # 562 imágenes
│   │   ├── train/labels/        # Anotaciones YOLO
│   │   ├── val/images/
│   │   └── val/labels/
│   ├── pillar.yaml              # Config dataset YOLO
│   ├── video.mp4                # Video de prueba
│   ├── video_frames/            # Frames extraídos
│   └── templates/               # Templates para auto-anotación
├── scripts/
│   ├── train.py                 # Entrenamiento completo
│   ├── inference.py             # Inferencia (imagen/video/webcam)
│   ├── auto_annotate.py         # Auto-anotación con template matching
│   ├── visualize_annotations.py # Visualizar anotaciones
│   ├── review_annotations.py    # Revisor interactivo de anotaciones
│   ├── split_dataset.py         # Dividir train/val
│   ├── evaluate.py              # Evaluación de métricas
│   ├── export_tensorrt.py       # Exportación a TensorRT/ONNX
│   └── benchmark.py             # Comparar velocidad de formatos
├── models/                      # Modelos (.pt, .onnx, .engine)
└── runs/                        # Logs de entrenamiento
```

---

## Instalación

```bash
# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Verificar CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Uso

### Entrenamiento

```bash
source .venv/bin/activate
python scripts/train.py
```

Configuración en `config.yaml`:
- `epochs`: Número de épocas (30-100 recomendado)
- `batch`: Tamaño de batch (8 para RTX 2060)
- `model`: Modelo base (`yolo12s.pt`)

### Inferencia

```bash
# Imagen
python scripts/inference.py --source imagen.jpg

# Video (usa TensorRT si está disponible)
python scripts/inference.py --source video.mp4 --model models/vr_boxes_best_20251207_171823.engine

# Webcam
python scripts/inference.py --source 0 --show

# Ajustar confianza (default: 0.65)
python scripts/inference.py --source video.mp4 --conf 0.5
```

### Exportar a TensorRT

```bash
# Exportar a TensorRT FP16 (2.5x más rápido)
python scripts/export_tensorrt.py --format engine --half

# Exportar a ONNX
python scripts/export_tensorrt.py --format onnx
```

### Benchmark

```bash
python scripts/benchmark.py
```

---

## Pipeline de Anotación

### 1. Extraer frames de video

```python
import cv2
from pathlib import Path

video = cv2.VideoCapture('data/video.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
frame_num = 0
saved = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    if frame_num % int(fps) == 0:  # 1 frame por segundo
        cv2.imwrite(f'data/frames/frame_{saved:04d}.jpg', frame)
        saved += 1
    frame_num += 1
video.release()
```

### 2. Auto-anotación con template matching

```bash
python scripts/auto_annotate.py \
  --frames data/frames/ \
  --template data/templates/pillar.jpg \
  --output data/labels/ \
  --threshold 0.7
```

**Limitaciones encontradas:**
- Template matching funciona bien con objetos distintivos
- Para patrones repetitivos (tablero amarillo/negro), genera falsos positivos
- Threshold 0.85-0.95 reduce falsos positivos pero puede perder detecciones

### 3. Anotación manual (cuando template matching falla)

Usamos **makesense.ai** (web, gratis):
1. Subir imágenes
2. Crear clase "pillar"
3. Dibujar bounding boxes
4. Exportar en formato YOLO

**Problema encontrado:** LabelImg crashea en Python 3.12 con PyQt5. Makesense.ai es la alternativa más estable.

### 4. Visualizar anotaciones

```bash
python scripts/visualize_annotations.py \
  --images data/frames/ \
  --labels data/labels/ \
  --output data/viz/ \
  --sample 10
```

---

## Problemas Encontrados y Soluciones

### 1. Template matching con falsos positivos

**Problema:** El patrón de tablero amarillo/negro es muy repetitivo, generando muchas detecciones falsas.

**Solución:**
- Aumentar threshold a 0.85-0.95
- Para pilares con patrón diferente (invertido), anotar manualmente
- Separar frames por tipo de pilar y procesar por separado

### 2. LabelImg crashea en Ubuntu/Python 3.12

**Error:** `TypeError: drawLine(): argument 1 has unexpected type 'float'`

**Causa:** Bug conocido de LabelImg con PyQt5 en Python 3.12

**Solución:** Usar makesense.ai (web) en su lugar

### 3. Pilares con patrón invertido no detectados

**Problema:** El modelo entrenado solo con pilares "amarillo arriba" no detecta pilares "negro arriba"

**Solución:**
1. Identificar frames con pilares invertidos (282-391)
2. Crear template específico para patrón invertido
3. Auto-anotar esos frames
4. Re-entrenar con datos ampliados

### 4. Dataset con anotaciones vacías

**Problema:** 92 imágenes tenían archivos .txt pero vacíos (sin anotaciones)

**Diagnóstico:**
```bash
wc -l data/dataset/train/labels/frame_000300.txt
# Output: 0  (vacío)
```

**Solución:** Re-anotar esos frames específicos

### 5. Entorno virtual desaparece

**Problema:** El .venv dejó de funcionar misteriosamente

**Solución:** Recrear:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6. SSH sin display gráfico

**Problema:** `--show` no funciona por SSH

**Solución:**
- El video se guarda en `runs/inference/`
- Descargar con `scp` o usar `ssh -X` para X11 forwarding

---

## Historial de Entrenamientos

### Entrenamiento 1: YOLOv8n, 10 épocas
- Dataset: Solo pilares normales
- Resultado: mAP50-95 = 62.7%

### Entrenamiento 2: YOLOv12s, 50 épocas
- Dataset: Solo pilares normales
- Resultado: mAP50-95 = 84.7%
- **Mejora significativa**

### Entrenamiento 3: YOLOv12s, 100 épocas
- Dataset: + 7 frames con pilares invertidos (manual)
- Resultado: mAP50-95 = 87.4%
- Los pilares invertidos siguen sin detectarse (pocos ejemplos)

### Entrenamiento 4: YOLOv12s, 30 épocas (en curso)
- Dataset: + 92 frames con pilares invertidos (auto-anotados)
- Objetivo: Detectar ambos tipos de pilares

---

## Comparativa YOLO8 vs YOLO12

| Métrica | YOLOv8n (10 ep) | YOLOv12s (50 ep) |
|---------|-----------------|------------------|
| Precision | 90.4% | 93.3% |
| Recall | 93.0% | 97.5% |
| mAP50 | 96.4% | 97.9% |
| mAP50-95 | 62.7% | 84.7% |

**Conclusión:** YOLOv12s con más épocas mejora significativamente la localización (mAP50-95).

---

## Recursos de GPU (RTX 2060)

| Operación | VRAM | Potencia |
|-----------|------|----------|
| Idle | ~400MB | ~10W |
| Entrenamiento | ~3.5GB | ~115W |
| Inferencia | ~1.5GB | ~50W |

Temperatura típica durante entrenamiento: 65-70°C

---

## Próximos Pasos

- [ ] Verificar detección de pilares invertidos tras entrenamiento 4
- [ ] Exportar modelo a TensorRT para inferencia más rápida
- [ ] Crear script de benchmark comparativo
- [ ] Probar con más videos de prueba

---

## Referencias

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [YOLOv12](https://docs.ultralytics.com/models/yolo12/)
- [makesense.ai](https://www.makesense.ai/) - Anotación web gratuita
- [Formato YOLO](https://docs.ultralytics.com/datasets/detect/)

---

*Última actualización: Diciembre 2025*
