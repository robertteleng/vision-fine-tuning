# Roadmap — Fine-Tuning Studio

Transformacion del proyecto de detector VR-especifico a framework universal de fine-tuning.

---

## Estado General

| Fase | Descripcion | Estado |
|------|-------------|--------|
| 0 | Renombrar directorio local | Completado |
| 1 | `src/hardware.py` — Auto-deteccion de GPU | Completado |
| 2 | `src/project.py` — Sistema de proyectos | Completado |
| 3 | Reescribir `config.yaml` | Completado |
| 4 | Reescribir `src/training.py` | Completado |
| 5 | Reescribir `src/inference.py` | Completado |
| 6 | Reescribir `src/annotation.py` | Completado |
| 7 | Reescribir `src/dataset.py` y `src/benchmark.py` | Completado |
| 8 | Reescribir `app.py` (Fine-Tuning Studio) | Completado |
| 9 | Reescribir `scripts/` | Completado |
| 10 | Reescribir `tests/` | Completado |
| 11 | Reescribir documentacion | Completado |
| 12 | Centralizar `find_best_model()` | Completado |

---

## Fase 0: Renombrar directorio local — Completado

- Repo renombrado en GitHub: `robertteleng/fine-tuning`
- Remote actualizado
- Directorio local renombrado

---

## Fase 1: `src/hardware.py` — Completado

- `detect_gpu()` — GPUInfo dataclass con nombre, VRAM, CUDA
- `suggest_training_defaults()` — batch/workers/cache/imgsz segun VRAM
- `get_hardware_summary()` — Resumen para UI
- Elimina todas las refs hardcoded a GPUs

---

## Fase 2: `src/project.py` — Completado

- Registro YOLO: v8, v11, v12, 26 x (n,s,m,l,x) x (detect,segment,classify,pose,obb)
- `ProjectConfig` dataclass con `from_yaml()`
- `get_models_for_task(task)` y `get_task_for_model(name)`
- `find_dataset_yamls()` para auto-discovery

---

## Fase 3: Reescribir `config.yaml` — Completado

- Eliminadas refs a VR/pillar/hardware especifico
- Default: `yolo26m.pt`, `batch: -1` (auto), `task: detect`
- Clases genericas
- Path de proyecto relativo (`runs/train`)

---

## Fase 4: Reescribir `src/training.py` — Completado

- Metricas leidas dinamicamente de `results.csv`
- Dataset YAML como parametro (no hardcoded)
- Integra `src/hardware.py` para GPU info
- Lee params de `config.yaml` automaticamente

---

## Fase 5: Reescribir `src/inference.py` — Completado

- Mensajes genericos, sin refs VR
- `find_best_model()` canonico (TensorRT > PyTorch > ONNX)
- `find_available_models()` centralizado

---

## Fase 6: Reescribir `src/annotation.py` — Completado

- Clase derivada del prompt dinamicamente (no hardcoded "pillar")

---

## Fase 7: Reescribir `src/dataset.py` y `src/benchmark.py` — Completado

- `get_dataset_stats()` con distribucion de clases
- `benchmark.py` importa `find_available_models` de inference

---

## Fase 8: Reescribir `app.py` — Completado

- Titulo: "Fine-Tuning Studio"
- Selector de tarea (detect/segment/classify/pose/obb)
- Selector dinamico de modelos segun tarea
- Hardware auto-detectado en UI
- Sin refs VR/pillar

---

## Fase 9: Reescribir `scripts/` — Completado

- Naming generico en todos los scripts
- Auto-detect dataset YAML con `find_dataset_yamls()`
- `find_best_model()` importado de `src.inference`
- Docstrings y ejemplos actualizados

---

## Fase 10: Reescribir `tests/` — Completado

- Tests para `src/hardware.py` y `src/project.py`
- Test de no-hardcoding: verifica que no quedan refs VR/pillar
- 15 tests passing, 2 skipped (sin models/ y sin Grounding DINO)

---

## Fase 11: Reescribir documentacion — Completado

- README.md universal
- HITOS.md con todas las fases completadas
- Docs VR archivados en `docs/archive/`

---

## Fase 12: Centralizar `find_best_model()` — Completado

- `find_best_model()` canonico en `src/inference.py`
- Scripts importan de ahi en vez de duplicar

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
