# Outline Completo del Proyecto - YOLO VR Box Detection

## 1. ¿Qué estamos construyendo?

Un sistema que detecta **cajas con patrón de tablero (amarillo-negro)** en un entorno de realidad virtual. Es para un sistema de navegación asistida que ayuda a evitar colisiones.

```
[Cámara VR] → [Modelo YOLO] → [Detecta cajas] → [Alerta al usuario]
```

---

## 2. Flujo de trabajo completo

```
FASE 1: PREPARAR DATOS
┌─────────────────────────────────────────────────────────────┐
│  1. Capturar frames de Unreal Engine (ya tienes video.mp4) │
│  2. Crear template de la caja (ya tienes pillar_template)  │
│  3. Auto-anotar con template matching → auto_annotate.py   │
│  4. Verificar anotaciones visualmente → visualize_annotations.py │
│  5. Dividir dataset 80/20 → split_dataset.py               │
└─────────────────────────────────────────────────────────────┘
                              ↓
FASE 2: ENTRENAR MODELO
┌─────────────────────────────────────────────────────────────┐
│  6. Configurar dataset → data/cajas.yaml                   │
│  7. Entrenar YOLO → train.py (ya existe)                   │
│  8. Evaluar métricas → evaluate.py                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
FASE 3: OPTIMIZAR Y DESPLEGAR
┌─────────────────────────────────────────────────────────────┐
│  9. Exportar a TensorRT → export_tensorrt.py               │
│  10. Comparar velocidades → benchmark.py                   │
│  11. Usar en producción → inference.py (ya existe)         │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Los 7 scripts que vamos a crear

| # | Script | ¿Qué hace? | Entrada | Salida |
|---|--------|------------|---------|--------|
| 1 | `auto_annotate.py` | Encuentra cajas automáticamente usando el template | Frames + Template | Archivos .txt con coordenadas |
| 2 | `split_dataset.py` | Divide imágenes en train/val | Frames anotados | Carpetas train/ y val/ |
| 3 | `evaluate.py` | Mide qué tan bueno es el modelo | Modelo entrenado | Métricas (mAP, precision, recall) |
| 4 | `visualize_annotations.py` | Muestra las cajas dibujadas | Frames + anotaciones | Imágenes con rectángulos |
| 5 | `export_tensorrt.py` | Convierte modelo para GPU rápida | Modelo .pt | Modelo .engine |
| 6 | `benchmark.py` | Compara velocidades | Modelos | Tabla de tiempos |
| 7 | `cajas.yaml` | Dice a YOLO dónde está el dataset | - | Config file |

---

## 4. Conceptos clave (simplificados)

### Template Matching
```
Tienes una imagen pequeña de la caja (template)
La deslizas por toda la imagen grande
Donde "encaja bien" = detección
```

### Formato YOLO
```
Cada imagen tiene un archivo .txt con las cajas:
clase x_centro y_centro ancho alto

Ejemplo: 0 0.5 0.5 0.1 0.2
         │  │   │   │   └── alto (20% de la imagen)
         │  │   │   └────── ancho (10% de la imagen)
         │  │   └────────── centro Y (mitad vertical)
         │  └────────────── centro X (mitad horizontal)
         └───────────────── clase 0 (nuestra caja)
```

### NMS (Non-Maximum Suppression)
```
Problema: El template encuentra la misma caja 5 veces
Solución: NMS elimina duplicados, deja solo la mejor
```

---

## 5. Orden de implementación

```
Hito 1: auto_annotate.py           ← EMPEZAMOS AQUÍ
Hito 2: visualize_annotations.py
Hito 3: split_dataset.py
Hito 4: cajas.yaml
Hito 5: evaluate.py
Hito 6: export_tensorrt.py
Hito 7: benchmark.py
```

---

## 6. Estructura del proyecto

```
fine-tuning-vr/
├── config.yaml              # Configuración de entrenamiento
├── requirements.txt         # Dependencias Python
├── diary.md                 # Diario de desarrollo
├── data/
│   ├── pillar_template.jpg  # Template de la caja
│   ├── video.mp4            # Video fuente
│   ├── cajas.yaml           # Config dataset (por crear)
│   ├── train/               # Datos entrenamiento
│   ├── valid/               # Datos validación
│   └── test/                # Datos test
├── scripts/
│   ├── train.py             # Entrenamiento (existe)
│   ├── inference.py         # Inferencia (existe)
│   ├── auto_annotate.py     # Por crear
│   ├── split_dataset.py     # Por crear
│   ├── evaluate.py          # Por crear
│   ├── visualize_annotations.py  # Por crear
│   ├── export_tensorrt.py   # Por crear
│   └── benchmark.py         # Por crear
├── docs/
│   ├── PROJECT_OUTLINE.md   # Este archivo
│   ├── DOCUMENTATION.md     # Documentación técnica
│   └── START_PROMPT.md      # Especificación original
├── models/                  # Modelos guardados
├── notebooks/               # Jupyter notebooks
└── runs/                    # Resultados de entrenamiento
```

---

## 7. Hardware

| Fase | GPU | Batch Size | Tiempo estimado |
|------|-----|------------|-----------------|
| Actual | RTX 2060 (6GB) | 8 | ~5 min/100 epochs |
| Futuro | RTX 5060 Ti (16GB) | 32 | ~2 min/100 epochs |

---

## 8. Métricas objetivo

| Métrica | Mínimo | Bueno | Excelente |
|---------|--------|-------|-----------|
| mAP@0.5 | 0.70 | 0.85 | 0.90+ |
| Recall | 0.80 | 0.85 | 0.90+ |
| Precision | 0.85 | 0.90 | 0.95+ |
| Latencia | <30ms | <15ms | <5ms |
