# Roadmap — Fine-Tuning Studio

Transformacion del proyecto de detector VR-especifico a framework universal de fine-tuning.

---

## Estado General

| Fase | Descripcion | Estado |
|------|-------------|--------|
| 0 | Renombrar directorio local | Completado |
| 1 | `src/hardware.py` — Auto-deteccion de GPU | Pendiente |
| 2 | `src/project.py` — Sistema de proyectos | Pendiente |
| 3 | Reescribir `config.yaml` | Completado |
| 4 | Reescribir `src/training.py` | Pendiente |
| 5 | Reescribir `src/inference.py` | Pendiente |
| 6 | Reescribir `src/annotation.py` | Pendiente |
| 7 | Reescribir `src/dataset.py` y `src/benchmark.py` | Pendiente |
| 8 | Reescribir `app.py` (Fine-Tuning Studio) | Pendiente |
| 9 | Reescribir `scripts/` | Pendiente |
| 10 | Reescribir `tests/` | Pendiente |
| 11 | Reescribir documentacion | Completado |
| 12 | Centralizar `find_best_model()` | Pendiente |

---

## Fase 0: Renombrar directorio local — Completado

- Repo renombrado en GitHub: `robertteleng/fine-tuning`
- Remote actualizado
- Directorio local renombrado

---

## Fase 1: `src/hardware.py` — Auto-deteccion de GPU

Nuevo modulo para detectar GPU automaticamente y sugerir defaults optimos.

**Funciones:**
- `detect_gpu()` — Retorna info de la GPU (nombre, VRAM, CUDA)
- `suggest_training_defaults()` — Sugiere batch, workers, cache, imgsz segun GPU
- `get_hardware_summary()` — Resumen legible para mostrar en UI

**Elimina:** Todas las referencias hardcoded a GPUs especificas.

---

## Fase 2: `src/project.py` — Sistema de proyectos

Sistema de configuracion por proyecto con registro de modelos YOLO.

**Funciones:**
- Registro de modelos: YOLO v8, v11, v12, 26 x (n,s,m,l,x) x (detect,segment,classify,pose,obb)
- `ProjectConfig` dataclass con dataset, modelo, hiperparametros
- `get_models_for_task(task)` — Lista modelos disponibles para una tarea
- Directorio `projects/` para almacenar configuraciones

---

## Fase 3: Reescribir `config.yaml` — Completado

- Eliminadas refs a VR/pillar/hardware especifico
- Default: `yolo26m.pt`, `batch: -1` (auto), `task: detect`
- Clases genericas
- Path de proyecto relativo (`runs/train`)

---

## Fase 4: Reescribir `src/training.py`

- Eliminar metricas hardcoded
- Eliminar dataset paths hardcoded
- Leer metricas dinamicamente de `results.csv`
- Integrar auto-deteccion de GPU de `src/hardware.py`

---

## Fase 5: Reescribir `src/inference.py`

- Cambiar mensajes especificos a genericos
- Soporte multi-tarea (detect/segment/classify/pose/obb)
- Centralizar `find_best_model()` (usado en 4 scripts)

---

## Fase 6: Reescribir `src/annotation.py`

- Eliminar class names hardcoded
- Clases derivadas del prompt dinamicamente

---

## Fase 7: Reescribir `src/dataset.py` y `src/benchmark.py`

- Agregar analisis de distribucion de clases
- `benchmark.py` ya es casi generico, ajustes menores

---

## Fase 8: Reescribir `app.py` — Fine-Tuning Studio

- Titulo: "Fine-Tuning Studio"
- Selector de tarea (detect/segment/classify/pose/obb)
- Selector dinamico de modelos segun tarea
- Display de hardware auto-detectado
- Eliminar refs VR/pillar

---

## Fase 9: Reescribir `scripts/`

- Eliminar naming especifico -> naming generico
- Auto-detect dataset YAML
- Actualizar docstrings y ejemplos

---

## Fase 10: Reescribir `tests/`

- Eliminar tests especificos de datasets hardcoded
- Agregar `tests/test_hardware.py`
- Agregar `tests/test_project.py`

---

## Fase 11: Reescribir documentacion — Completado

- README.md universal
- HITOS.md como roadmap de fases
- Docs VR-especificos archivados en `docs/archive/`
- `ZERO_SHOT_GUIDE.md` conservado (ya es generico)

---

## Fase 12: Centralizar `find_best_model()`

`find_best_model()` esta duplicada en 4 scripts. Mover a `src/inference.py` como funcion canonica e importar en los demas.

---

## Auditoria de Hardcoding

| Archivo | Refs VR/Pillar | Refs Hardware | Total |
|---------|---------------|---------------|-------|
| app.py | 12+ | 5+ | 17+ |
| config.yaml | 8+ | 20+ | 28+ |
| scripts/train.py | 6+ | 5+ | 11+ |
| src/training.py | 5+ | 2+ | 7+ |
| src/annotation.py | 2 | 0 | 2 |
| src/inference.py | 1 | 0 | 1 |
| tests/test_integration.py | 4 | 0 | 4 |

**Totales:**
- 40+ refs a "pillar", "vr_box", "VR Pillar Detector"
- 20+ refs a hardware especifico
- 12+ metricas hardcoded
- 15+ paths hardcoded
- 6+ class names/counts hardcoded

---

## Specs Tecnicos

### YOLO26
- Ultralytics >=8.4.14
- Naming: `yolo26[n/s/m/l/x][-seg/-cls/-pose/-obb].pt`
- NMS-free, end-to-end
- MuSGD optimizer, ProgLoss, STAL
- 43% mas rapido en CPU que v11

### Hardware Target
- GPU: NVIDIA RTX 5060 Ti (16GB VRAM)
- Defaults optimos: batch=-1 (auto), workers=8, cache="ram", amp=true

---

*Ultima actualizacion: Febrero 2026*
