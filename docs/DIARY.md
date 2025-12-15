# Diario de Desarrollo - YOLO VR Box Detection

## Resumen del Proyecto
Fine-tuning de YOLO para detectar cajas con patrón tablero amarillo-negro en VR para navegación asistida.

---

## 2025-12-03 - Inicio del Desarrollo

### Estado Inicial del Proyecto
**Archivos existentes:**
- `config.yaml` - Configuración de entrenamiento
- `requirements.txt` - Dependencias
- `scripts/train.py` - Script de entrenamiento
- `scripts/inference.py` - Script de inferencia
- `docs/DOCUMENTATION.md` - Documentación técnica
- `docs/START_PROMPT.md` - Especificación del proyecto
- `data/` - Dataset con train/valid/test splits, template y video

### Hito 1: auto_annotate.py [COMPLETADO]
- Script de auto-anotación con template matching multi-escala y NMS
- Template: 280x544 px
- Imágenes procesadas: 694
- Total detecciones: 2102
- Velocidad: ~3.4 img/s

---

## 2025-12-07 - Sesión de Desarrollo Intensivo

### Problema 1: Template Matching con Falsos Positivos

**Síntoma:** El auto_annotate.py generaba muchas detecciones falsas debido al patrón repetitivo del tablero amarillo/negro.

**Diagnóstico:**
- Threshold bajo (0.7) detectaba cualquier zona con contraste amarillo/negro
- El patrón del tablero se repetía en múltiples zonas de la imagen

**Solución:**
- Aumentar threshold a 0.85-0.95 para reducir falsos positivos
- Para casos complejos, usar anotación manual con makesense.ai

---

### Problema 2: LabelImg Crashea en Python 3.12

**Síntoma:** Al intentar usar LabelImg para anotar manualmente, crasheaba con error:
```
TypeError: drawLine(): argument 1 has unexpected type 'float'
```

**Diagnóstico:** Bug conocido de LabelImg con PyQt5 en Python 3.12.

**Solución:** Usar makesense.ai (aplicación web) en lugar de LabelImg.

---

### Problema 3: Pilares Invertidos No Detectados

**Síntoma:** El modelo entrenado solo con pilares "amarillo arriba" no detectaba pilares con patrón invertido "negro arriba".

**Diagnóstico:**
- Entrenamiento 1 y 2 solo tenían pilares normales
- Entrenamiento 3 añadió 7 frames manuales - insuficiente para aprender el patrón

**Solución:**
1. Identificar frames con pilares invertidos (282-391)
2. Crear template específico para patrón invertido
3. Auto-anotar esos frames con nuevo template
4. Crear `review_annotations.py` para revisar anotaciones manualmente
5. Resultado final: 59 frames con pilares invertidos añadidos y validados

---

### Problema 4: Dataset con Anotaciones Vacías

**Síntoma:** 92 imágenes tenían archivos .txt vacíos (0 bytes).

**Diagnóstico:**
```bash
wc -l data/dataset/train/labels/frame_000300.txt
# Output: 0  (vacío)
```

**Solución:** Re-anotar frames específicos mediante revisión manual.

---

### Problema 5: Ghost Detections (Detecciones Fantasma)

**Síntoma:** El modelo detectaba objetos inexistentes con confianza 0.5-0.55.

**Diagnóstico:** El threshold de confianza por defecto (0.25) era demasiado bajo.

**Solución:** Cambiar threshold de confianza por defecto a 0.65 en `scripts/inference.py`:
```python
# Antes
default=0.25

# Después
default=0.65
```

---

### Problema 6: TensorRT - No Space Left on Device

**Síntoma:** Al intentar instalar TensorRT con pip, fallaba con:
```
ERROR: Could not install packages due to an OSError:
[Errno 28] No space left on device
```

**Diagnóstico:**
```bash
df -h /tmp
# /tmp estaba en partición root de 26GB con solo 4GB libres
# TensorRT necesita ~4GB para descomprimir durante instalación

df -h /home
# /home tenía 810GB disponibles
```

**Solución:** Redirigir directorio temporal a /home:
```bash
export TMPDIR=~/tmp
mkdir -p $TMPDIR
pip install tensorrt
```

**Resultado:** Instalación exitosa. TensorRT ocupó ~4GB en .venv.

---

### Problema 7: Entorno Virtual Corrupto

**Síntoma:** El .venv dejó de funcionar después de múltiples instalaciones.

**Solución:** Recrear el entorno:
```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Problema 8: SSH sin Display Gráfico

**Síntoma:** `--show` en inference.py no funcionaba por SSH.

**Solución:**
- El video se guarda automáticamente en `runs/inference/`
- Descargar con `scp` para visualizar
- O usar `ssh -X` para X11 forwarding

---

## Hitos Completados

### Hito 2: split_dataset.py [COMPLETADO]
- División reproducible del dataset 80/20
- Implementado con seed para reproducibilidad

### Hito 3: evaluate.py [COMPLETADO]
- Evaluación de métricas mAP, precision, recall, F1
- Soporte para diferentes splits (train/val/test)

### Hito 4: visualize_annotations.py [COMPLETADO]
- Visualización de anotaciones para verificación
- Muestra bounding boxes sobre imágenes

### Hito 5: export_tensorrt.py [COMPLETADO]
- Exportación a TensorRT FP16
- También soporta ONNX, TorchScript, OpenVINO

### Hito 6: benchmark.py [COMPLETADO]
- Benchmark comparativo de formatos
- Resultados RTX 2060:
  - PyTorch: 14ms (71 FPS)
  - ONNX: 19.4ms (52 FPS)
  - TensorRT FP16: 5.5ms (180 FPS) - **2.5x más rápido**

### Hito 7: data/pillar.yaml [COMPLETADO]
- Configuración del dataset para YOLO
- Clase única: "pillar"

### Hito Extra: review_annotations.py [COMPLETADO]
- Revisor interactivo de anotaciones
- Teclas: A (anterior), D (siguiente), S (guardar), Q (salir)

---

## Historial de Entrenamientos

| # | Modelo | Épocas | Dataset | mAP50-95 | Observaciones |
|---|--------|--------|---------|----------|---------------|
| 1 | YOLOv8n | 10 | 555 train | 62.7% | Baseline |
| 2 | YOLOv12s | 50 | 555 train | 84.7% | +22 puntos |
| 3 | YOLOv12s | 100 | +7 invertidos | 87.4% | Insuficiente |
| **4** | **YOLOv12s** | **100** | **+59 invertidos** | **87.4%** | **FINAL** |

---

## Resultados Finales

| Métrica | Valor |
|---------|-------|
| mAP@50 | 98.7% |
| mAP@50-95 | 87.4% |
| Precisión | 91.7% |
| Recall | 99.2% |
| Inferencia PyTorch | 14ms (71 FPS) |
| **Inferencia TensorRT FP16** | **5.5ms (180 FPS)** |

---

## Limpieza Realizada

### Espacio Recuperado
- Runs de entrenamiento antiguos: ~4GB
- Carpetas temporales (frames_slim_*, video_frames_labels*, viz_check): ~250MB
- Modelos base en raíz (yolo11n.pt, yolo12s.pt): ~50MB

### Estructura Final
```
.venv/          14GB (TensorRT + dependencias)
runs/           1.9GB (solo último entrenamiento)
data/           194MB (dataset limpio)
models/         74MB (.pt, .onnx, .engine)
scripts/        92KB (9 scripts)
docs/           48KB (documentación)
```

---

## Notas y Decisiones

- Modelo final: YOLOv12s (mejor balance velocidad/precisión)
- Threshold de confianza: 0.65 para producción
- Formato recomendado: TensorRT FP16 para inferencia en tiempo real
- Hardware: RTX 2060 (6GB) - suficiente para entrenar y ejecutar

---

## Próximos Pasos

- [ ] Probar con más videos de prueba
- [ ] Comparativa cuando llegue RTX 5060 Ti
- [ ] Considerar data augmentation si se necesita más robustez

---

## 2025-12-15 - Auto-Anotación Zero-Shot

### Objetivo
Eliminar la necesidad de anotar manualmente usando modelos zero-shot que detectan objetos a partir de descripciones de texto.

### Experimento 1: YOLO-World (FALLIDO)

**Script:** `scripts/auto_annotate_zeroshot.py`

**Cómo funciona:**
- YOLO-World combina YOLO con CLIP para entender descripciones de texto
- Le dices "yellow black checkered box" y busca objetos que coincidan

**Resultados:**
| Prompt probado | Detecciones | Confianza |
|----------------|-------------|-----------|
| "yellow black checkered cube" | 0 | - |
| "checkered block" | 0 | - |
| "box" (genérico) | 6 | 0.05-0.09 |
| "yellow box" | 0 | - |

**Conclusión:** YOLO-World NO funciona para objetos VR específicos. CLIP fue entrenado con imágenes del mundo real y no reconoce bien patrones sintéticos/VR.

---

### Experimento 2: Grounding DINO (ÉXITO)

**Scripts:**
- `scripts/test_grounding_dino.py` - Test rápido para una imagen
- `scripts/auto_annotate_grounding_dino.py` - Pipeline completo

**Cómo funciona:**
- Grounding DINO usa un modelo de lenguaje más potente
- Mejor para objetos inusuales y prompts descriptivos
- Usa HuggingFace transformers (fácil de instalar)

**Resultados:**
```
Prompt: "yellow and black checkered box"
Threshold: 0.25

Total imágenes:      47
Con detecciones:     47 (100%)
Total detecciones:   259
Promedio/imagen:     5.51
Confianza:           0.25-0.69
```

**Comparativa:**
| Métrica | YOLO-World | Grounding DINO |
|---------|------------|----------------|
| Cobertura | 8.5% | 100% |
| Detecciones | 6 | 259 |
| Confianza | 0.05-0.09 | 0.25-0.69 |
| Velocidad | ~25 img/s | ~2 img/s |

**Conclusión:** Grounding DINO es **43x mejor** para nuestro caso.

---

### Uso de Grounding DINO

```bash
# Test rápido en una imagen
python scripts/test_grounding_dino.py \
    --image data/video_frames/frame_0010.jpg \
    --prompt "yellow and black checkered box" \
    --threshold 0.25

# Auto-anotar dataset completo
python scripts/auto_annotate_grounding_dino.py \
    --source data/video_frames/ \
    --prompt "yellow and black checkered box" \
    --output data/dataset_gdino/ \
    --threshold 0.25 \
    --class-name "vr_pillar"
```

---

### Documentación Creada

- `docs/ZERO_SHOT_GUIDE.md` - Guía educativa completa con:
  - Diagramas de flujo (Mermaid)
  - Explicaciones estilo pizarra
  - Guía de prompts con ejemplos
  - Troubleshooting visual
  - Cheatsheet de 1 página

---

### Datasets Actuales

| Dataset | Imágenes | Anotaciones | Origen |
|---------|----------|-------------|--------|
| `data/dataset/` | 701 | 219 | Template matching + manual |
| `data/dataset_gdino/` | 47 | 259 | Grounding DINO |
| `data/dataset_zeroshot_test*/` | - | - | **BORRAR** (pruebas) |

---

### Pendiente

- [ ] Limpiar directorios de prueba (`dataset_zeroshot_test*`)
- [ ] Decidir si combinar datasets para entrenar
- [ ] Anotar `video_what_frames/` (145 imágenes) con Grounding DINO
- [ ] Entrenar modelo con nuevo dataset

---

*Última actualización: 15 Diciembre 2025*
