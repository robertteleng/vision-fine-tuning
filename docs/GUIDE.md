# Guía de la Aplicación Web - VR Pillar Detector

## Inicio Rápido

```bash
# Activar entorno virtual
source .venv/bin/activate

# Iniciar la aplicación
python scripts/app.py

# Con enlace público (para compartir)
python scripts/app.py --share

# Puerto personalizado
python scripts/app.py --port 8080
```

La aplicación se abrirá automáticamente en tu navegador en `http://127.0.0.1:7860`

---

## Pestañas de la Aplicación

### 1. Inferencia

La pestaña principal para detectar pilares en imágenes y videos.

#### Inferencia en Imagen

1. **Subir imagen**: Arrastra o selecciona una imagen JPG/PNG
2. **Seleccionar modelo**: Elige entre PyTorch (.pt), ONNX (.onnx) o TensorRT (.engine)
3. **Ajustar parámetros**:
   - **Confianza mínima** (0.65 recomendado): Filtra detecciones con baja confianza
   - **IoU threshold** (0.45 recomendado): Controla el solapamiento entre detecciones
4. **Click en "Detectar"**: Ver resultados con bounding boxes

#### Inferencia en Video

1. **Subir video**: Formatos soportados MP4, AVI, MOV
2. **Configurar parámetros**: Igual que para imágenes
3. **Click en "Procesar Video"**: El procesamiento muestra una barra de progreso
4. **Descargar resultado**: El video procesado se puede descargar

**Nota**: El procesamiento de video puede tardar dependiendo de la duración y el modelo seleccionado.

---

### 2. Métricas

Visualiza las métricas del modelo y ejecuta benchmarks.

#### Métricas del Modelo

Muestra los resultados del entrenamiento:
- **mAP@50**: Precisión con IoU 0.5 (98.7%)
- **mAP@50-95**: Precisión promediando IoUs (87.4%)
- **Precisión**: Detecciones correctas / Total detecciones (91.7%)
- **Recall**: Objetos detectados / Total objetos reales (99.2%)

#### Benchmark en Vivo

Compara la velocidad de los modelos disponibles:

1. **Seleccionar iteraciones**: Más iteraciones = resultados más precisos
2. **Click en "Ejecutar Benchmark"**
3. **Ver resultados**: Velocidad en ms y FPS para cada formato

Resultados típicos en RTX 2060:
| Formato | Velocidad | FPS |
|---------|-----------|-----|
| TensorRT FP16 | 5.5 ms | 180 |
| PyTorch | 14 ms | 71 |
| ONNX | 19 ms | 52 |

---

### 3. Entrenamiento

Configura y lanza nuevos entrenamientos.

#### Parámetros

- **Épocas** (50-100 recomendado): Más épocas = mejor aprendizaje pero más tiempo
- **Batch Size** (8 para RTX 2060): Reducir si hay errores de memoria
- **Modelo Base**: YOLOv12s recomendado para balance velocidad/precisión

#### Proceso

1. Configurar parámetros
2. Click en "Iniciar Entrenamiento"
3. Esperar (aproximadamente 15-25 min para 50-100 épocas)
4. El modelo se guarda automáticamente en `runs/train/`

**Importante**: El entrenamiento usa la GPU al máximo. Evitar otras tareas pesadas.

---

### 4. Info

Información general del proyecto y enlaces útiles.

---

## Argumentos de Línea de Comandos

```bash
python scripts/app.py [opciones]

Opciones:
  --share       Crear enlace público (gradio.live)
  --port PORT   Puerto del servidor (default: 7860)
  --host HOST   Host del servidor (default: 127.0.0.1)
```

### Ejemplos

```bash
# Ejecutar localmente
python scripts/app.py

# Compartir públicamente (crea URL temporal)
python scripts/app.py --share

# Acceder desde otra máquina en la red local
python scripts/app.py --host 0.0.0.0

# Puerto específico
python scripts/app.py --port 8080 --host 0.0.0.0
```

---

## Solución de Problemas

### Error: "No se encontraron modelos"

Los modelos deben estar en la carpeta `models/`:
```bash
ls models/
# Debería mostrar: vr_boxes_best_*.pt, *.onnx, *.engine
```

### Error de CUDA / GPU no detectada

Verificar instalación de CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### La app no abre el navegador

Abrir manualmente: `http://127.0.0.1:7860`

### Video procesado sin audio

El procesamiento de video no preserva el audio. Es una limitación conocida.

### Entrenamiento muy lento

- Reducir batch size
- Cerrar otras aplicaciones
- Verificar que usa GPU: debería mostrar ~115W de consumo

---

## Acceso Remoto

### Por SSH

```bash
# En el servidor
python scripts/app.py --host 0.0.0.0 --port 7860

# En tu máquina local
ssh -L 7860:localhost:7860 usuario@servidor
# Luego abrir http://localhost:7860
```

### Enlace Público (Gradio Share)

```bash
python scripts/app.py --share
```

Esto crea un enlace temporal tipo `https://xxxxx.gradio.live` que funciona por 72 horas.

---

## Comparativa: App vs CLI

| Funcionalidad | App Web | CLI |
|---------------|---------|-----|
| Inferencia imagen | Fácil (drag & drop) | `python scripts/inference.py --source img.jpg` |
| Inferencia video | Con progreso visual | `python scripts/inference.py --source vid.mp4` |
| Webcam | No soportado | `python scripts/inference.py --source 0 --show` |
| Benchmark | Interactivo | `python scripts/benchmark.py` |
| Entrenamiento | Básico | `python scripts/train.py` (más opciones) |
| Exportar modelo | No | `python scripts/export_tensorrt.py` |

**Recomendación**: Usa la app para demos y pruebas rápidas. Usa CLI para trabajo en producción.

---

## Arquitectura

```
scripts/app.py
├── create_app()           # Crea la interfaz Gradio
├── run_inference_image()  # Inferencia en imágenes
├── run_inference_video()  # Inferencia en videos
├── run_benchmark()        # Benchmark de velocidad
├── start_training()       # Inicia entrenamiento
└── main()                 # Punto de entrada
```

La app usa:
- **Gradio 6.x**: Framework de interfaz web
- **Ultralytics YOLO**: Motor de detección
- **OpenCV**: Procesamiento de video
- **PyTorch/TensorRT**: Backend de inferencia

---

*Última actualización: Diciembre 2024*
