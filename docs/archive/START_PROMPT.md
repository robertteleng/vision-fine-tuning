# YOLO VR Box Detection - Project Setup Prompt

**Copy this entire prompt to GitHub Copilot / Claude Code to generate the complete project.**

---

## Project Goal

Create a complete YOLOv12 fine-tuning project for detecting yellow-black checkerboard boxes in VR for collision avoidance.

---

## Project Structure

```
yolo_vr_boxes/
├── README.md (comprehensive with quick start)
├── requirements.txt
├── config.yaml (training configuration)
├── .gitignore
├── data/
│   ├── template/ (box template image)
│   ├── frames/ (VR screenshots)
│   ├── annotations/ (YOLO .txt files)
│   ├── train/images/ and train/labels/
│   ├── val/images/ and val/labels/
│   └── cajas.yaml (dataset config)
├── scripts/
│   ├── auto_annotate.py (template matching)
│   ├── split_dataset.py (80/20 split)
│   ├── train.py (YOLOv12 training)
│   ├── evaluate.py (metrics & validation)
│   ├── inference.py (batch/realtime/benchmark)
│   ├── export_tensorrt.py (TensorRT optimization)
│   ├── benchmark.py (speed comparison)
│   └── visualize_annotations.py (check annotations)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── docs/
│   ├── unreal_capture.md
│   ├── template_creation.md
│   ├── benchmarks.md
│   └── tuning_guide.md
└── models/ (saved models)
```

---

## Requirements

### requirements.txt

```
ultralytics==8.3.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python==4.9.0.80
numpy==1.26.0
matplotlib==3.8.0
pyyaml==6.0.1
tqdm==4.66.1
labelImg==1.8.6
jupyter==1.0.0
notebook==7.0.0
onnx>=1.14.0
onnxruntime-gpu>=1.15.0
```

---

## Key Files

### config.yaml (Training Configuration)

```yaml
# Model
model: yolo12n.pt  # YOLOv12 nano (fastest)

# Dataset
data: data/cajas.yaml

# Training
epochs: 100
batch: 8  # RTX 2060 → 16-32 for RTX 5060 Ti
imgsz: 640

# Optimizer
optimizer: SGD
lr0: 0.01
momentum: 0.937

# Augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
translate: 0.1
scale: 0.5
flipud: 0.0
fliplr: 0.5
mosaic: 1.0

# Hardware
device: 0
workers: 4  # → 8 for RTX 5060 Ti
```

### data/cajas.yaml

```yaml
path: /absolute/path/to/yolo_vr_boxes/data  # UPDATE THIS
train: train/images
val: val/images
nc: 1
names:
  0: caja_checkerboard
```

---

## Script Implementations

### scripts/auto_annotate.py

**Purpose:** Auto-annotate frames using template matching

**Key features:**
- Multi-scale template matching (0.3× to 2.0×)
- Non-maximum suppression (NMS) to remove duplicates
- YOLO format output (.txt files)
- Progress bars with tqdm
- Statistics reporting

**Algorithm:**
1. Load template (grayscale)
2. For each frame:
   - Try 20 scales
   - Match template (cv2.TM_CCOEFF_NORMED)
   - Threshold at 0.65 (configurable)
   - Apply NMS with overlap_thresh=0.3
3. Save normalized YOLO coordinates

### scripts/split_dataset.py

**Purpose:** Split annotated data into 80/20 train/val

**Features:**
- Reproducible split (seed=42)
- Copy images + labels to train/val directories
- Statistics reporting

### scripts/train.py

**Purpose:** Train YOLOv12 model

**Features:**
- Load config from config.yaml
- Train with all hyperparameters
- Save timestamped model
- Print final metrics
- Clear status messages

### scripts/evaluate.py

**Purpose:** Evaluate trained model

**Features:**
- Calculate mAP, precision, recall, F1
- Speed metrics (ms/frame, FPS)
- Interpretation of results
- Visualization of predictions (first 10 val images)

### scripts/inference.py

**Purpose:** Run inference in 3 modes

**Modes:**
1. **batch**: Process directory of images
2. **realtime**: Live webcam/video feed with FPS overlay
3. **benchmark**: Speed testing (100 iterations)

**Features:**
- Configurable confidence threshold
- FPS calculation
- Statistics reporting

### scripts/export_tensorrt.py

**Purpose:** Export model to TensorRT for speed

**Features:**
- FP16 precision option
- Workspace configuration
- Verification after export

### scripts/benchmark.py

**Purpose:** Compare inference speed across formats

**Formats tested:**
- PyTorch (.pt)
- TensorRT FP32 (.engine)
- TensorRT FP16 (.engine)

---

## Documentation Files

### docs/unreal_capture.md

Explain:
- Blueprint setup for automatic screenshot capture
- Timer configuration (0.5s interval)
- Console command: HighResShot 1920x1080
- Capture strategy (distances, angles, scenarios)
- Expected output location

### docs/template_creation.md

Explain:
- How to select good source frame
- Cropping technique (one box, frontal, medium distance)
- Target size (~256×384 pixels)
- Quality checklist
- Tools: GIMP, Photoshop, or Python/OpenCV

### docs/benchmarks.md

Template for tracking:
- Hardware specs
- Training time per epoch
- Inference latency
- mAP scores
- Comparison RTX 2060 vs 5060 Ti

### docs/tuning_guide.md

Explain how to adjust:
- Template matching threshold
- NMS overlap threshold
- Training hyperparameters
- Augmentation strength
- When to use each adjustment

---

## README.md Structure

1. **Overview** (project goal, features)
2. **Hardware Requirements** (current + upgrade)
3. **Quick Start** (10 steps from setup to deployment)
4. **Project Workflow** (flowchart)
5. **Configuration** (key parameters to adjust)
6. **Troubleshooting** (common issues + solutions)
7. **Performance Benchmarks** (table)
8. **Integration** (Scene Aria System example code)
9. **Documentation** (links to other docs)
10. **License & Acknowledgments**

---

## Critical Implementation Details

### Auto-annotation NMS Algorithm

```python
def nms_boxes(detections, overlap_thresh=0.3):
    # Convert to numpy
    boxes = np.array([[d['x'], d['y'], d['x']+d['w'], d['y']+d['h']] 
                      for d in detections])
    scores = np.array([d['conf'] for d in detections])
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Calculate IOU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        order = order[np.where(iou <= overlap_thresh)[0] + 1]
    
    return [detections[i] for i in keep]
```

### Training Loop

```python
# Load model
model = YOLO(config['model'])

# Train
results = model.train(
    data=config['data'],
    epochs=config['epochs'],
    batch=config['batch'],
    # ... all config parameters
)

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f'models/yolo12n_cajas_{timestamp}.pt')
```

### Real-time Inference

```python
model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Inference
    start = time.time()
    results = model(frame, conf=0.5, verbose=False)
    fps = 1 / (time.time() - start)
    
    # Draw results
    annotated = results[0].plot()
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow('Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## Code Quality Requirements

1. **Type hints** in all function signatures
2. **Docstrings** (Google style) with examples
3. **Error handling** with clear messages
4. **Progress bars** (tqdm) for long operations
5. **Status messages** at each major step
6. **Comments** explaining non-obvious code
7. **PEP 8** compliance

---

## Testing Checklist

Generated project must:
- [ ] Install without errors
- [ ] Have complete README
- [ ] Include all scripts (runnable)
- [ ] Have proper .gitignore
- [ ] Include config with comments
- [ ] Provide clear error messages
- [ ] Include .gitkeep in empty dirs

---

## Generate Complete Project

Create ALL files with:
- Complete implementations
- Extensive comments
- Clear documentation
- Production-ready code
- Educational value

**Start generating now!**
`