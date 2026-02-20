# Auto-Anotacion con Grounding DINO

## Que es Grounding DINO

Grounding DINO es un modelo zero-shot de Google/IDEA-Research que detecta objetos a partir de una descripcion textual. No necesita entrenamiento previo — le describes lo que buscas y lo encuentra.

## Por Que Grounding DINO y No Etiquetar a Mano

| Metodo | Tiempo para 500 imagenes | Calidad |
|--------|--------------------------|---------|
| Manual (makesense.ai) | 8-16 horas | Alta (humano) |
| Grounding DINO | 5-15 minutos | Buena (>80% precision) |
| YOLO-World | 2-5 minutos | Aceptable (>70% precision) |

Grounding DINO es mejor que YOLO-World porque usa un modelo de lenguaje mas potente para entender los prompts.

## Flujo de Trabajo

```
1. Recopilar imagenes
2. Escribir prompt en ingles
3. Ejecutar auto_annotate_grounding_dino.py
4. Revisar visualmente (UI o visualize_annotations.py)
5. Corregir errores si los hay
6. Listo para entrenar
```

## Como Funciona Internamente

```
Texto: "staircase steps"
  ↓
Tokenizer → Text Encoder (BERT) → Text Features
  ↓
Imagen 640x640
  ↓
Image Backbone (Swin Transformer) → Image Features
  ↓
Cross-Attention: Text Features × Image Features
  ↓
Decoder → Bounding Boxes + Scores
  ↓
Filtrar por threshold → Labels YOLO
```

## Guia de Prompts

### Reglas de Oro

1. **Siempre en ingles** — el modelo esta entrenado en ingles
2. **Formula**: `[color] + [material/patron] + [objeto]`
3. **4-5 palabras maximo** — prompts largos confunden al modelo
4. **Punto al final** — Grounding DINO lo requiere (el script lo añade automaticamente)

### Ejemplos para Navegacion Asistida

| Objeto | Prompt bueno | Prompt malo | Por que |
|--------|-------------|-------------|---------|
| Escaleras | `staircase steps` | `stairs` | Mas especifico |
| Puerta | `door entrance` | `door` | Contexto de acceso |
| Paso peatones | `pedestrian crosswalk zebra crossing` | `crosswalk` | Describe el patron |
| Bordillo | `sidewalk curb edge` | `curb` | Contexto de acera |
| Poste | `street pole lamppost` | `pole` | Tipo especifico |
| Banco | `public bench seat` | `bench` | Diferencia de otros bancos |

### Ajustar Threshold

| Threshold | Efecto | Usar cuando |
|-----------|--------|-------------|
| 0.15-0.25 | Muchas detecciones, mas falsos positivos | Objetos dificiles de detectar |
| 0.25-0.35 | Balance bueno | Default recomendado |
| 0.35-0.50 | Pocas detecciones, muy precisas | Objetos obvios y grandes |

## Comando Detallado

```bash
python scripts/auto_annotate_grounding_dino.py \
    --source data/frames/              # Carpeta con imagenes
    --prompt "staircase steps"         # Descripcion del objeto
    --output data/dataset_stairs/      # Carpeta de salida
    --threshold 0.3                    # Umbral de confianza
    --class-name stairs                # Nombre de clase (opcional)
    --val-split 0.2                    # 20% para validacion
    --visualize 10                     # Guardar 10 previews
    --model IDEA-Research/grounding-dino-tiny  # Modelo (tiny o base)
    --copy-images                      # Copiar en vez de symlinks
```

## Estructura de Salida

```
data/dataset_stairs/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── val/
│       └── img_003.jpg
├── labels/
│   ├── train/
│   │   ├── img_001.txt    # "0 0.52 0.34 0.23 0.45"
│   │   └── img_002.txt
│   └── val/
│       └── img_003.txt
├── visualizations/
│   ├── viz_img_001.jpg    # Preview con bboxes dibujados
│   └── viz_img_002.jpg
└── dataset.yaml           # Config YOLO
```

## Requisitos

```bash
pip install transformers torch torchvision
pip install opencv-python Pillow tqdm pyyaml
```

El modelo se descarga automaticamente la primera vez (~350MB para tiny).

## Multi-Clase

Para detectar multiples tipos de objetos, hay dos estrategias:

### Estrategia A: Una ejecucion por clase

```bash
# Ejecutar separado para cada clase
python scripts/auto_annotate_grounding_dino.py \
    --source data/frames/ --prompt "staircase" \
    --output data/nav/ --class-name stairs

python scripts/auto_annotate_grounding_dino.py \
    --source data/frames/ --prompt "door entrance" \
    --output data/nav/ --class-name door
```

Despues editar `dataset.yaml` para incluir todas las clases.

### Estrategia B: Script multi-clase personalizado

El script actual solo soporta una clase por ejecucion. Para multi-clase automatico, se puede extender el script para aceptar multiples prompts y asignar clase 0, 1, 2... a cada uno.

## Troubleshooting

| Problema | Solucion |
|----------|----------|
| No detecta nada | Bajar threshold a 0.15, cambiar prompt |
| Muchos falsos positivos | Subir threshold a 0.4, prompt mas especifico |
| Modelo lento en CPU | Usar GPU (`--device cuda`) |
| OOM en GPU | Usar `grounding-dino-tiny` en vez de `base` |
| Bounding boxes cortados | Verificar que objetos no esten en bordes |
