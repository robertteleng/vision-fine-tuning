# YOLO VR Box Detection - Complete Technical Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Theoretical Foundations](#theoretical-foundations)
3. [Complete Step-by-Step Workflow](#complete-step-by-step-workflow)
4. [Technical Deep Dives](#technical-deep-dives)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Performance Optimization](#performance-optimization)
7. [References & Resources](#references--resources)

---

## 1. PROJECT OVERVIEW

### 1.1 Problem Statement

Detect yellow-black checkerboard patterned boxes in a VR environment for collision avoidance in an assistive navigation system (Scene Aria System with Meta Aria glasses).

**Key Requirements:**
- Real-time detection (>30 FPS)
- High recall (>85%) - missing obstacles is dangerous
- Low latency (<15ms per frame)
- Robust to varying distances and angles

### 1.2 Why YOLO for This Task

**YOLO (You Only Look Once) advantages:**
- Single-shot detection: one forward pass = all detections
- Real-time performance: optimized for speed
- Good accuracy-speed tradeoff
- Pre-trained on COCO: transfer learning ready

**YOLOv12 specifically:**
- Latest architecture (2025)
- Improved accuracy over v11
- Better small object detection
- Optimized inference pipeline

### 1.3 Hardware Setup

**Current (Phase 1):**
- GPU: NVIDIA RTX 2060 (6GB VRAM)
- CPU: Intel NUC
- Batch size: 8
- Expected training time: 5 minutes (100 epochs)
- Inference: ~12ms per frame

**Upgrade (Phase 2 - in 2 weeks):**
- GPU: NVIDIA RTX 5060 Ti (16GB VRAM)
- Batch size: 32
- Expected training time: 2 minutes (100 epochs)
- Inference: ~4ms per frame

### 1.4 Success Metrics

**Detection Performance:**
- mAP@0.5: >0.85 (good), >0.90 (excellent)
- Precision: >0.90 (few false alarms)
- Recall: >0.85 (critical - don't miss obstacles)

**Speed Performance:**
- Inference latency: <15ms (RTX 2060), <5ms (5060 Ti)
- End-to-end pipeline: <30ms total
- Target FPS: >30 (real-time navigation)

---

## 2. THEORETICAL FOUNDATIONS

### 2.1 Object Detection Fundamentals

**Object Detection vs Classification:**

| Task | Input | Output | Example |
|------|-------|--------|---------|
| Classification | Image | Class label | "This is a box" |
| Detection | Image | Class + location | "Box at (x,y,w,h)" |
| Segmentation | Image | Pixel-wise mask | Every pixel labeled |

**Bounding Box Representation:**

```
Standard format (x, y, w, h):
- x, y: top-left corner pixel coordinates
- w, h: width and height in pixels

YOLO format (normalized):
- x_center: (x + w/2) / image_width
- y_center: (y + h/2) / image_height  
- width: w / image_width
- height: h / image_height

All values in range [0.0, 1.0]
```

**Example conversion:**

```python
# Image: 1920x1080 pixels
# Box: top-left (500, 300), size 200x150

# Calculate center
x_center = (500 + 200/2) / 1920 = 600/1920 = 0.3125
y_center = (300 + 150/2) / 1080 = 375/1080 = 0.3472

# Normalize dimensions
width = 200 / 1920 = 0.1042
height = 150 / 1080 = 0.1389

# YOLO format: class x_center y_center width height
# Result: 0 0.3125 0.3472 0.1042 0.1389
```

**Intersection over Union (IOU):**

```
IOU = Area of Overlap / Area of Union

Example:
Box A: [100, 100, 50, 50] (x, y, w, h)
Box B: [120, 120, 50, 50]

Intersection area: 30×30 = 900
Union area: (50×50) + (50×50) - 900 = 4100
IOU = 900 / 4100 = 0.22

IOU > 0.5 typically considered "good match"
```

**Mean Average Precision (mAP):**

```
For each class:
1. Sort all predictions by confidence (high to low)
2. For each prediction, check if IOU > threshold
3. Calculate precision at each recall level
4. Average precision (AP) = area under precision-recall curve

mAP = mean(AP) across all classes

mAP@0.5 = mAP with IOU threshold 0.5
mAP@0.5:0.95 = average mAP from IOU 0.5 to 0.95 (step 0.05)
```

### 2.2 Transfer Learning & Fine-tuning

**Pre-training on COCO:**

YOLOv12 comes pre-trained on COCO dataset:
- 330,000 images
- 80 common object classes
- Learned features:
  - Low-level: edges, corners, textures
  - Mid-level: shapes, curves, patterns
  - High-level: object concepts (cars, people, etc.)

**Why Transfer Learning Works:**

1. **Feature Hierarchy:**
   - Low-level features (edges) are universal
   - Mid-level features (rectangles) apply to boxes
   - Only high-level features need adjustment

2. **Data Efficiency:**
   - Training from scratch: need 10,000+ images
   - Fine-tuning: need 200-500 images
   - Our case: 250 images sufficient

3. **Faster Convergence:**
   - Random init: 500+ epochs to converge
   - Pre-trained: 50-100 epochs sufficient

### 2.3 Template Matching

**How cv2.matchTemplate Works:**

Template matching = sliding window + correlation

```python
# For each position (x, y) in image:
#   1. Extract window same size as template
#   2. Calculate similarity score
#   3. Store score in result matrix

Methods (cv2.TM_CCOEFF_NORMED):
- Normalized cross-correlation
- Range: [-1, 1]
- 1 = perfect match
- 0 = no correlation
- -1 = perfect inverse
```

**Multi-scale Detection:**

Problem: Template is fixed size, but boxes appear at different distances

Solution: Resize template, try all scales

```python
for scale in [0.3, 0.4, 0.5, ..., 1.8, 1.9, 2.0]:
    resized_template = resize(template, scale)
    result = matchTemplate(image, resized_template)
    # Find peaks above threshold
```

**Non-Maximum Suppression (NMS):**

Problem: Multiple detections of same box at nearby scales/positions

```
Original detections:
Box 1: (100, 100, 50, 50) conf=0.9
Box 2: (102, 101, 52, 51) conf=0.85  ← duplicate of Box 1
Box 3: (300, 200, 60, 60) conf=0.92

NMS algorithm:
1. Sort by confidence: [Box 3, Box 1, Box 2]
2. Keep Box 3 (highest)
3. Keep Box 1 (doesn't overlap Box 3)
4. Remove Box 2 (overlaps Box 1 by >30%)

Final: [Box 3, Box 1]
```

---

## 3. COMPLETE STEP-BY-STEP WORKFLOW

### Step 1: Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics==8.3.0
pip install opencv-python==4.9.0.80
pip install numpy==1.26.0
pip install matplotlib==3.8.0
pip install labelImg==1.8.6

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 2: Unreal Engine Frame Capture

**Blueprint Implementation:**

```
Event BeginPlay
  → Set Timer by Function Name
      Function Name: "CaptureFrame"
      Time: 0.5
      Looping: True

Custom Event: CaptureFrame
  → Format Text: "frame_{0}" (FrameCounter padded to 6 digits)
  → Execute Console Command: "HighResShot 1920x1080 filename=..."
  → Increment FrameCounter
```

**Capture Strategy:**
- Straight approach: 100 frames (walk toward boxes)
- Angled approaches: 80 frames (45° left/right, 30° up/down)
- Peripheral vision: 40 frames (boxes at edge of view)
- Multiple boxes: 30 frames (2-3 boxes visible)

**Total: ~250 frames**

### Step 3: Template Creation

1. Open `frame_000027.jpg` (any frame with clear box)
2. Crop ONE complete box:
   - Include full checkerboard (2×4 pattern)
   - Minimal background
   - Frontal view
   - Size: ~256×384 pixels
3. Save as `data/template/box_template.jpg`

**Quality Checklist:**
- ✓ Full pattern visible
- ✓ No partial squares cut off
- ✓ Minimal background (<5%)
- ✓ Good contrast
- ✓ No blur

### Step 4: Auto-annotation with Template Matching

```bash
python scripts/auto_annotate.py
```

**Algorithm:**
1. Load template
2. For each frame:
   - Try 20 different scales (0.3× to 2.0×)
   - Find all matches above threshold (0.65)
   - Apply NMS to remove duplicates
   - Save YOLO format (.txt)

**Expected output:**
```
Total frames: 250
Total detections: 487
Average boxes/frame: 1.95
```

### Step 5: Validation with LabelImg

```bash
pip install labelImg
cd data
labelimg frames/ annotations/
```

**Check 20+ frames for:**
- ✓ All visible boxes annotated?
- ✓ No false positives?
- ✓ Boxes tight around object?
- ✓ No duplicates?

### Step 6: Dataset Split

```bash
python scripts/split_dataset.py
```

Creates 80/20 train/val split:
- Train: 200 images
- Val: 50 images

### Step 7: Training

```bash
python scripts/train.py
```

**Config highlights:**
- Model: yolo12n.pt
- Epochs: 100
- Batch: 8 (RTX 2060)
- Augmentations: HSV, translate, scale, mosaic

**Expected time:** ~5 minutes

**Target metrics:**
- mAP@0.5: >0.85
- Recall: >0.85
- Precision: >0.90

### Step 8: Evaluation

```bash
python scripts/evaluate.py
```

Reviews:
- Overall mAP, precision, recall
- Speed metrics (ms/frame, FPS)
- Per-class performance
- Visualizations of predictions

### Step 9: Inference

```bash
# Batch inference
python scripts/inference.py --source data/val/images/ --save

# Real-time
python scripts/inference.py --mode realtime --source 0

# Benchmark
python scripts/inference.py --mode benchmark
```

### Step 10: Optimization

```bash
# Export to TensorRT (2-3× speedup)
python scripts/export_tensorrt.py

# Benchmark all formats
python scripts/benchmark.py
```

---

## 4. TROUBLESHOOTING GUIDE

### Issue: Low mAP (<0.70)

**Solutions:**
1. Re-validate dataset in LabelImg (fix annotation errors)
2. Capture 100+ more diverse frames
3. Train longer (200 epochs)
4. Use larger model (yolo12s.pt)

### Issue: Low Recall (<0.80) - CRITICAL

**This is most important for safety!**

**Solutions:**
1. Lower confidence threshold: `conf=0.3` in inference
2. Check which boxes are missed (visualize predictions)
3. Add more training data with missed scenarios
4. Increase box loss weight in config

### Issue: Slow Inference (>30ms)

**Solutions:**
1. Export to TensorRT + FP16
2. Optimize preprocessing (resize before YOLO)
3. Reduce confidence threshold (fewer boxes to process)
4. Use asynchronous inference

### Issue: High False Positives

**Solutions:**
1. Raise confidence threshold: `conf=0.7`
2. Hard negative mining (add false positive frames to training)
3. Temporal filtering for video streams

---

## 5. PERFORMANCE OPTIMIZATION

### TensorRT Export

```python
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='engine', half=True, device=0)

# Load TensorRT model
model_trt = YOLO('best.engine')

# Expected speedup: 2-3×
```

### Inference Optimization

```python
# Enable all optimizations
results = model(
    frame,
    half=True,          # FP16 precision
    device=0,           # GPU
    verbose=False,      # No printing
    conf=0.5,           # Confidence threshold
    iou=0.7,            # NMS threshold
    augment=False       # No TTA
)
```

### GPU Upgrade Impact (RTX 2060 → 5060 Ti)

| Metric | RTX 2060 | RTX 5060 Ti | Improvement |
|--------|----------|-------------|-------------|
| Inference | 12ms | 4ms | 3.0× |
| Training | 5 min | 2 min | 2.5× |
| Max batch | 8 | 32 | 4.0× |
| FPS | 80 | 240 | 3.0× |

---

## 6. REFERENCES & RESOURCES

### Core Papers

1. "You Only Look Once: Unified, Real-Time Object Detection"  
   Redmon et al., CVPR 2016

2. "Focal Loss for Dense Object Detection"  
   Lin et al., ICCV 2017

### Documentation

- Ultralytics YOLO: https://docs.ultralytics.com/
- OpenCV: https://docs.opencv.org/
- PyTorch: https://pytorch.org/docs/

### Tools

- LabelImg: https://github.com/HumanSignal/labelImg
- TensorRT: https://developer.nvidia.com/tensorrt
- Roboflow: https://roboflow.com/

---

**Document Version:** 1.0  
**Last Updated:** 2024-12-03  
**Project:** YOLO VR Box Detection  
**Model:** YOLOv12  
**Hardware:** RTX 2060 → RTX 5060 Ti
