# Caso de Uso: Navegacion Asistida con YOLO + Meta Aria

## El Problema

Personas con discapacidad visual necesitan detectar obstaculos y elementos de navegacion en tiempo real para moverse de forma segura por entornos urbanos e interiores.

## La Solucion

Un modelo YOLO26 fine-tuned que corre en edge (gafas Meta Aria Gen 2 o dispositivo con GPU) detectando multiples clases de obstaculos simultaneamente.

## Por Que YOLO26 + TensorRT

| Caracteristica | Valor |
|----------------|-------|
| Latencia | 3.02ms por frame (TensorRT FP16) |
| FPS | 331 en RTX 5060 Ti |
| Modelo | yolo26m.pt — 22M parametros |
| Peso TensorRT | 41 MB |
| NMS-free | Si — end-to-end, sin post-procesado |

A 331 FPS, el modelo puede procesar video en tiempo real sin ningun problema. Incluso en hardware mas limitado (Jetson, movil) se mantendria por encima de 30 FPS.

## Clases de Navegacion Propuestas

Objetos criticos para navegacion urbana e interior:

| Clase | Prompt Grounding DINO | Prioridad |
|-------|----------------------|-----------|
| stairs_up | `upward staircase steps` | Alta |
| stairs_down | `downward staircase steps` | Alta |
| door | `door entrance` | Alta |
| crosswalk | `pedestrian crosswalk zebra crossing` | Alta |
| curb | `sidewalk curb edge` | Media |
| pole | `street pole lamppost` | Media |
| bench | `public bench seat` | Baja |
| bollard | `bollard post barrier` | Media |
| construction | `construction barrier warning` | Alta |
| bicycle | `parked bicycle` | Baja |

## Meta Aria Gen 2 — Contexto Tecnico

Las gafas Meta Aria Gen 2 (disponibles Q2 2026) ofrecen:

- **Camara HDR 120dB** — funciona en condiciones de luz extremas
- **SLAM 6DOF** — tracking espacial en tiempo real
- **Eye tracking** — saber donde mira el usuario
- **Hand tracking** — interaccion gestual
- **Audio espacial** — feedback por sonido posicional

El pipeline seria:
```
Camara Aria → Modelo YOLO (edge/cloud) → Audio espacial
                                          "Escaleras a 3 metros a la derecha"
```

## Referencias

- [Meta Aria Gen 2](https://www.projectaria.com/glasses/)
- [YOLO-OD: Obstacle Detection for Visually Impaired](https://www.mdpi.com/1424-8220/24/23/7621)
- [Investigating YOLO Models for Outdoor Obstacle Detection](https://arxiv.org/html/2312.07571v1)
- [Envision + Aria: Indoor Navigation for Blind](https://ai.meta.com/blog/aria-gen-2-research-glasses-under-the-hood-reality-labs/)
