# Benchmarks y Métricas de Entrenamiento

Este documento registra las métricas de entrenamiento y comparativas de rendimiento.

---

## Métricas de Referencia

### Métricas YOLO Explicadas

| Métrica | Descripción | Valor Ideal |
|---------|-------------|-------------|
| **mAP@50** | Mean Average Precision con IoU 0.5 | > 0.90 |
| **mAP@50-95** | mAP promediando IoUs de 0.5 a 0.95 | > 0.70 |
| **Precision** | Detecciones correctas / Total detecciones | > 0.90 |
| **Recall** | Detecciones correctas / Total objetos reales | > 0.85 |

---

## Historial de Entrenamientos

### Entrenamiento 1: 2025-12-07 - YOLOv8n Baseline

#### Configuración
- **Modelo base**: yolov8n.pt
- **GPU**: NVIDIA RTX 2060 (6GB VRAM)
- **Batch size**: 8
- **Image size**: 640
- **Épocas**: 10
- **Dataset**: 555 train / 139 val

#### Métricas Finales
| Métrica | Valor |
|---------|-------|
| Precision | 90.4% |
| Recall | 93.0% |
| mAP@50 | 96.4% |
| mAP@50-95 | 62.7% |

#### Tiempos
- Tiempo total: ~3 min
- Tiempo por época: ~18s
- Inferencia: ~5ms/imagen

#### Observaciones
- Primer entrenamiento de baseline
- Solo pilares normales (amarillo arriba)
- mAP50-95 bajo indica localización imprecisa

---

### Entrenamiento 2: 2025-12-07 - YOLOv12s 50 épocas

#### Configuración
- **Modelo base**: yolo12s.pt
- **GPU**: NVIDIA RTX 2060 (6GB VRAM)
- **Batch size**: 8
- **Image size**: 640
- **Épocas**: 50
- **Dataset**: 555 train / 139 val

#### Métricas Finales
| Métrica | Valor |
|---------|-------|
| Precision | 93.3% |
| Recall | 97.5% |
| mAP@50 | 97.9% |
| mAP@50-95 | 84.7% |

#### Tiempos
- Tiempo total: ~15 min
- Tiempo por época: ~18s
- Inferencia: ~4ms/imagen

#### Observaciones
- Mejora significativa en mAP50-95 (+22 puntos)
- YOLOv12s mejor que YOLOv8n para este caso
- Pilares invertidos siguen sin detectarse

---

### Entrenamiento 3: 2025-12-07 - YOLOv12s 100 épocas + 7 frames invertidos

#### Configuración
- **Modelo base**: yolo12s.pt
- **GPU**: NVIDIA RTX 2060 (6GB VRAM)
- **Batch size**: 8
- **Image size**: 640
- **Épocas**: 100
- **Dataset**: 562 train / 139 val (+7 frames manuales)

#### Métricas Finales
| Métrica | Valor |
|---------|-------|
| Precision | 91.7% |
| Recall | 99.2% |
| mAP@50 | 98.7% |
| mAP@50-95 | 87.4% |

#### Tiempos
- Tiempo total: ~24 min (0.4h)
- Tiempo por época: ~14s
- Inferencia: ~3.5ms/imagen

#### Observaciones
- 7 frames con pilares invertidos añadidos manualmente
- No suficiente para aprender el nuevo patrón
- El modelo aún no detecta pilares invertidos
- Error de scipy al final (gráficos), pero modelo guardado correctamente

---

### Entrenamiento 4: 2025-12-07 - YOLOv12s FINAL con pilares invertidos

#### Configuración
- **Modelo base**: yolo12s.pt
- **GPU**: NVIDIA RTX 2060 (6GB VRAM)
- **Batch size**: 8
- **Image size**: 640
- **Épocas**: 100
- **Dataset**: 621 train / 139 val (pilares normales + 59 frames invertidos revisados manualmente)

#### Métricas Finales
| Métrica | Valor |
|---------|-------|
| Precision | 91.7% |
| Recall | 99.2% |
| mAP@50 | 98.7% |
| mAP@50-95 | 87.4% |

#### Tiempos
- Tiempo total: ~24 min
- Tiempo por época: ~14s
- Inferencia PyTorch: 14ms/imagen
- Inferencia TensorRT FP16: 5.5ms/imagen

#### Observaciones
- 59 frames con pilares invertidos añadidos tras revisión manual
- Detecta correctamente ambos tipos de pilares
- Confianza threshold: 0.65 para filtrar ghost detections
- Modelo exportado a TensorRT FP16

---

## Comparativa de Modelos

| Entrenamiento | Modelo | Épocas | mAP50 | mAP50-95 | Inferencia |
|---------------|--------|--------|-------|----------|------------|
| #1 | YOLOv8n | 10 | 96.4% | 62.7% | 5ms |
| #2 | YOLOv12s | 50 | 97.9% | 84.7% | 4ms |
| #3 | YOLOv12s | 100 | 98.7% | 87.4% | 3.5ms |
| **#4 FINAL** | YOLOv12s | 100 | 98.7% | 87.4% | **5.5ms TRT** |

**Conclusión:** YOLOv12s supera a YOLOv8n significativamente en mAP50-95, lo que indica mejor localización de bounding boxes.

---

## Recursos de GPU (RTX 2060)

| Operación | VRAM | Potencia | Temperatura |
|-----------|------|----------|-------------|
| Idle (desktop) | ~500MB | ~10W | ~45°C |
| Entrenamiento YOLO12s | ~3.5GB | ~115W | ~67°C |
| Entrenamiento + Desktop | ~4.3GB | ~117W | ~67°C |
| Inferencia | ~1.5GB | ~50W | ~55°C |

**Nota:** Con entorno de escritorio activo, el consumo de VRAM y potencia aumenta ~200MB y ~5W respectivamente.

---

## Configuración Óptima RTX 2060

```yaml
model: yolo12s.pt
batch: 8
imgsz: 640
workers: 4
epochs: 50-100
patience: 20
amp: true
```

**Tips:**
- No exceder batch=8 para evitar OOM
- AMP (mixed precision) activado por defecto
- 50+ épocas recomendadas para buenos resultados
- Cerrar aplicaciones pesadas durante entrenamiento

---

## Benchmark de Inferencia por Formato

| Formato | Velocidad | FPS | Speedup | Tamaño |
|---------|-----------|-----|---------|--------|
| PyTorch (.pt) | 14.0 ms | 71 | 1x | 18 MB |
| ONNX (.onnx) | 19.4 ms | 52 | 0.7x | 35.5 MB |
| **TensorRT FP16 (.engine)** | **5.5 ms** | **180** | **2.5x** | 21 MB |

**Conclusión:** TensorRT FP16 es 2.5x más rápido que PyTorch nativo.

---

## Próximos Experimentos

- [x] ~~Evaluar resultados con frames invertidos~~ COMPLETADO
- [x] ~~Exportar a TensorRT~~ COMPLETADO
- [ ] Probar con más videos de prueba
- [ ] Comparativa cuando llegue RTX 5060 Ti

---

*Última actualización: 7 Diciembre 2025*
