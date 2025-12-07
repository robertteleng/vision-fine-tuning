# VR Pillar Detector

Detection of yellow/black signaling pillars in Virtual Reality environments.

## Requirements

- **Python 3.10+** - [Download](https://www.python.org/downloads/)
- **NVIDIA GPU** (optional, for faster inference)

## Installation (Windows)

1. Install Python 3.10+ (check "Add Python to PATH")
2. Double-click `install.bat`
3. Wait for dependencies to install

## Usage

1. Double-click `run.bat`
2. Browser opens at `http://localhost:7860`

## Features

| Tab | Description |
|-----|-------------|
| **Inference** | Detect pillars in images and videos |
| **Metrics** | View model performance and run benchmarks |
| **Training** | Start new training sessions |
| **Annotations** | Review and edit dataset annotations |
| **Info** | Project information and CLI usage |

## Model Performance

| Metric | Value |
|--------|-------|
| mAP@50 | 98.7% |
| Precision | 91.7% |
| Recall | 99.2% |
| Speed (GPU) | ~70 FPS |
| Speed (CPU) | ~5 FPS |

## Troubleshooting

### "Python is not installed"
Download and install Python from https://www.python.org/downloads/
Make sure to check "Add Python to PATH" during installation.

### "CUDA not available"
The app works without CUDA, but slower. For GPU acceleration:
1. Install NVIDIA drivers
2. Install CUDA Toolkit 11.8+

### App doesn't open in browser
Manually open: http://localhost:7860

## Command Line Options

```bash
# Custom port
.venv\Scripts\python app.py --port 8080

# Share publicly (creates temporary URL)
.venv\Scripts\python app.py --share

# Allow network access
.venv\Scripts\python app.py --host 0.0.0.0
```

## License

MIT License
