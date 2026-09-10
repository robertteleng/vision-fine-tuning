# Lab 07 · FP32 → FP16 → INT8 en la RTX 5060 Ti

> Lab para hacer **tú**, paso a paso y tomando notas. La IA puede ayudarte, pero escalando la ayuda: pregunta → pista → señalar el error → solución, solo al final (ver `learning/robotics-interview-prep/METODO.md`).
> Las soluciones de los ejercicios a mano están al final. No las mires antes de intentarlo.

## Objetivo observable

Al terminar tienes dos cosas:

1. **Una tabla medida en tu máquina, con comandos reproducibles**: 3 precisiones × latencia (media, p50, p95), tamaño del engine, mAP50, mAP50-95 y mAP50 de **Stairs**, **Door** y **Person**.
2. **Poder explicar sin mirar** qué es la cuantización INT8, por qué necesita calibración y cuándo no merece la pena.

## Criterio de decisión (escríbelo ANTES de medir)

Esto es un ejemplo; ajústalo y fírmalo con fecha antes del paso 3.

> Usaré INT8 en vez de FP16 solo si se cumplen las tres condiciones:
> - la p95 baja al menos un 20 %;
> - el mAP50 global cae 0.01 como máximo;
> - el mAP50 de Stairs cae 0.02 como máximo.

Decidir el criterio después de ver los números es la forma más fácil de engañarte.

---

## Paso 0 · Entorno (10 min)

Anota estos datos. Sin ellos, ningún número de este lab es reproducible.

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run python -c "import torch, ultralytics; print(torch.__version__, torch.version.cuda, ultralytics.__version__)"
uv run python -c "import tensorrt; print(tensorrt.__version__)"
git rev-parse --short HEAD
```

**Verifica en el código fuente de tu versión**, no en la documentación web, que puede ir por delante:

```bash
ULT=$(uv run python -c "import ultralytics, os; print(os.path.dirname(ultralytics.__file__))")
grep -n "int8\|quantize\|fraction" $ULT/cfg/default.yaml
grep -n "CALIBRATION\|fraction\|split" $ULT/engine/exporter.py | head -30
```

Preguntas para tus notas:

1. ¿El argumento de INT8 se llama `int8=True` o `quantize=8` en tu versión? La doc actual usa `quantize` y marca `int8` como obsoleto; tu lockfile tiene ultralytics 8.4.14.
2. ¿Qué calibrador usa, MinMax o Entropy?
3. **¿Con qué split calibra?** Si calibra con `val` y luego evalúas con `val`, ¿contaminas la evaluación? Razónalo antes de seguir.

## Paso 1 · Entender: conceptos (escribe tu definición en 1–2 líneas y después contrástala)

| Concepto | Tu definición |
|---|---|
| FP32 / FP16 / INT8 (cómo se representa un número) | |
| Rango dinámico | |
| Cuantización: **scale** y **zero-point** | |
| Simétrica vs asimétrica | |
| Per-tensor vs per-channel | |
| **PTQ** (post-training quantization) | |
| **QAT** (quantization-aware training) | |
| **Calibración** y dataset de calibración | |
| Calibradores: MinMax, Entropy (KL), Percentile | |
| Error de redondeo vs error de *clipping* | |
| Outliers en activaciones | |
| Cuantización implícita vs explícita (nodos Q/DQ) | |
| Precisión mixta / *fallback* de capas a FP16 | |
| Layer fusion y kernel auto-tuning | |
| Engine (plan) no portable y calibration cache | |
| Latencia vs throughput; media vs p95 | |
| Warmup y `torch.cuda.synchronize()` | |
| mAP50 vs mAP50-95 | |

## Paso 2 · Simular a mano (sin ordenador)

Activaciones de una capa: `[0.01, 0.03, 0.12, 0.7, 5.8]`. Cuantización INT8 simétrica per-tensor: `q = round(x / scale)`, recortado a [-127, 127]; para volver, `x' = q · scale`.

**A.** Calcula `scale = max|x| / 127` (MinMax). Calcula `q` y `x'` para cada valor, y el error `|x − x'|`.

**B.** Ahora recorta el rango a 1.0 (lo que haría un calibrador tipo percentil): `scale = 1.0 / 127`. Repite los cálculos.

**C.** Responde:
- ¿Qué valores salen ganando en A y cuáles en B?
- La mayoría de activaciones reales se parecen a los valores pequeños. ¿Qué calibrador elegirías y qué riesgo corres?
- ¿Por qué un solo outlier puede hundir la precisión de una clase pequeña como Stairs?

**Predicciones** (escríbelas ahora, antes del paso 3):

| | Mi predicción |
|---|---|
| Latencia FP16 vs FP32 (×) | |
| Latencia INT8 vs FP16 (×) | |
| Caída de mAP50 global con INT8 | |
| Caída de mAP50 de Stairs con INT8 | |
| Tamaño del engine INT8 vs FP16 | |

## Paso 3 · Escribir: ejecutar paso a paso

Usa el mismo `data`, `imgsz=640`, `batch=1` y `device=0` en todo. **Ojo:** cada export sobrescribe `yolo26s_nav.engine`, así que renómbralo justo después de exportar.

```bash
# 3.1 · Baseline FP32 (PyTorch)
uv run yolo val model=models/yolo26s_nav.pt data=data/nav_combined/dataset.yaml imgsz=640 batch=1 device=0

# 3.2 · FP16 (usa quantize=16 si tu versión lo soporta)
uv run yolo export model=models/yolo26s_nav.pt format=engine half=True imgsz=640 batch=1 device=0
mv models/yolo26s_nav.engine models/yolo26s_nav_fp16.engine
uv run yolo val model=models/yolo26s_nav_fp16.engine data=data/nav_combined/dataset.yaml imgsz=640 batch=1 device=0

# 3.3 · INT8 con calibración (usa quantize=8 si tu versión lo soporta)
#   Recomendado: un yaml de calibración cuyo 'val' apunte a ≥500 imágenes de TRAIN (ver pregunta 3 del paso 0)
uv run yolo export model=models/yolo26s_nav.pt format=engine int8=True data=data/nav_calib.yaml imgsz=640 batch=1 device=0
mv models/yolo26s_nav.engine models/yolo26s_nav_int8.engine
uv run yolo val model=models/yolo26s_nav_int8.engine data=data/nav_combined/dataset.yaml imgsz=640 batch=1 device=0
```

**Ejercicio de código (intenta tú primero):** amplía `scripts/benchmark.py` para que:

1. Use **N imágenes reales** de validación en vez de ruido aleatorio.
2. Reporte **p50, p95 y p99** además de la media.
3. Guarde un JSON con las versiones, la GPU, el commit, el comando y los resultados.

Mientras corre, observa la VRAM con `nvidia-smi dmon -s um -d 1`.

## Paso 4 · Medir

| Precisión | Media (ms) | p50 | p95 | Engine (MB) | VRAM | mAP50 | mAP50-95 | Stairs | Door | Person |
|---|---|---|---|---|---|---|---|---|---|---|
| FP32 .pt | | | | | | | | | | |
| FP16 .engine | | | | | | | | | | |
| INT8 .engine | | | | | | | | | | |

Compara con tus predicciones del paso 2. ¿Dónde fallaste y por qué?

## Paso 5 · Mirar atrás

- ¿Cumple INT8 el criterio que firmaste? Decide: **FP16 o INT8**.
- Si INT8 perdió demasiado, estas son las palancas, de menos a más coste:
  - más imágenes de calibración, o más representativas;
  - otro calibrador;
  - dejar en FP16 las capas sensibles (precisión mixta);
  - QAT.
- Si INT8 **no** fue mucho más rápido que FP16, busca por qué. Hipótesis a comprobar: modelo pequeño, capas que no se cuantizan o coste de conversión entre precisiones.
- Actualiza la tabla sin fuente "FP16 vs FP32" de `06-tensorrt-optimizacion.md` con **tus** números, citando el comando y el commit.

## Notas (una entrada por paso)

```
Fecha · Paso · Comando
Esperado:
Obtenido:
Hipótesis:
Lección:
```

## Preguntas por responder (sin IA, al final)

1. ¿Qué diferencia hay entre PTQ y QAT, y cuándo elegirías cada una?
2. ¿Para qué sirve la calibración? ¿Qué pasa si las imágenes de calibración no representan el dominio real?
3. ¿Por qué un `.engine` no es portable entre GPUs?
4. INT8 casi no fue más rápido que FP16. Da tres causas posibles.
5. ¿Por qué no basta con mirar el mAP global para aprobar un modelo cuantizado?

---

## Soluciones del paso 2 (mirar solo después)

<details><summary>Ver</summary>

**A (MinMax, clip 5.8):** `scale = 0.04567`
- `q = [0, 1, 3, 15, 127]`
- `x' = [0.0, 0.0457, 0.137, 0.685, 5.8]`
- error ≈ `[0.010, 0.016, 0.017, 0.015, 0.0]`

Los valores pequeños pierden casi toda su resolución: 0.01 se convierte en 0.

**B (clip 1.0):** `scale = 0.00787`
- `q = [1, 4, 15, 89, 127]`
- `x' = [0.0079, 0.0315, 0.1181, 0.7008, 1.0]`
- error ≈ `[0.002, 0.002, 0.002, 0.001, 4.8]`

Los valores pequeños ganan unas 8 veces de precisión, pero el outlier se recorta y pierde 4.8.

**C:** es un trade-off entre error de redondeo (A) y error de clipping (B). Si el outlier lleva información importante, recortarlo rompe detecciones. Si es ruido, recortar mejora el resto. Por eso la calibración usa datos reales y por eso hay que medir por clase, no solo el global.

</details>
