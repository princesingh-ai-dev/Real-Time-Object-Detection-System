# 👁️ Real-Time Object Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Real-time object detection system with YOLOv8/v10, GPU-accelerated inference, and a professional PyQt6 monitoring dashboard.**

Detects 80+ object classes in real-time while displaying live system performance metrics — CPU, RAM, GPU utilization & temperature.

</div>

---

## ✨ Features

- 🎯 **Multi-Model Support** — Switch between YOLOv8n, YOLOv8s, YOLOv10n, and YOLOv10s on the fly
- ⚡ **GPU Accelerated** — CUDA + TensorRT optimized for NVIDIA GPUs (tested on RTX 2050)
- 📊 **Live Performance Dashboard** — Real-time CPU, RAM, GPU utilization & temperature monitoring
- 🎚️ **Accuracy Modes** — Fast, Balanced, and Accurate modes with configurable confidence thresholds
- 📷 **Multi-Camera Support** — Auto-scan and switch between available cameras
- 🖥️ **Professional GUI** — Dark-themed PyQt6 interface with object detection overlays
- 🔍 **Object Info Panel** — Live object count with per-class detection details
- 🏎️ **TensorRT Export** — Export models to TensorRT for maximum inference speed

## 🎬 Architecture

```
┌──────────────┐     ┌───────────────┐     ┌────────────────────────┐
│   Camera     │────▶│  YOLO Model   │────▶│    PyQt6 GUI           │
│   Input      │     │  (GPU/CUDA)   │     │  ┌──────────────────┐  │
└──────────────┘     └───────────────┘     │  │ Video Panel      │  │
                            │              │  ├──────────────────┤  │
                            ▼              │  │ Object Info      │  │
                     ┌───────────────┐     │  ├──────────────────┤  │
                     │  Detections   │────▶│  │ Performance      │  │
                     │  + FPS Stats  │     │  │ Monitor (GPU/CPU)│  │
                     └───────────────┘     │  └──────────────────┘  │
                                           └────────────────────────┘
```

## 📋 Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended) or CPU
- Webcam
- Windows / Linux

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/princesingh1702/Real-Time-Object-Detection-System.git
cd Real-Time-Object-Detection-System

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run with GUI
python gui.py

# Or run headless detection
python run_detection.py
```

## 📁 Project Structure

```
Real-Time-Object-Detection-System/
├── gui.py                  # PyQt6 GUI with performance dashboard
├── detector.py             # YOLO detection engine (GPU-accelerated)
├── config.py               # Detection configuration & model settings
├── run_detection.py        # Headless detection runner
├── export_tensorrt.py      # TensorRT model export script
├── export_optimized.py     # Optimized model export
├── check_gpu.py            # GPU availability checker
├── test_gpu_monitoring.py  # GPU monitoring tests
├── test_trt_speed.py       # TensorRT speed benchmarks
├── requirements.txt        # Python dependencies
├── yolov8n.pt              # YOLOv8 Nano model
├── yolov8s.pt              # YOLOv8 Small model
├── yolov10n.pt             # YOLOv10 Nano model
└── yolov10s.pt             # YOLOv10 Small model
```

## ⚙️ Configuration

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| Model | `yolov8n`, `yolov8s`, `yolov10n`, `yolov10s` | `yolov8n` | Detection model |
| Accuracy Mode | `fast`, `balanced`, `accurate` | `balanced` | Speed vs accuracy tradeoff |
| Confidence | `0.1 – 1.0` | `0.5` | Minimum detection confidence |
| Camera | `0, 1, 2...` | `0` | Camera index |
| GPU | `CUDA`, `CPU` | Auto-detect | Inference device |

## 🏎️ Performance

| Model | Device | FPS | mAP@0.5 |
|-------|--------|-----|---------|
| YOLOv8n | RTX 2050 | ~45 FPS | 37.3% |
| YOLOv8s | RTX 2050 | ~35 FPS | 44.9% |
| YOLOv8s (TensorRT) | RTX 2050 | ~60 FPS | 44.9% |
| YOLOv10n | RTX 2050 | ~50 FPS | 38.5% |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Detection | Ultralytics YOLOv8 / YOLOv10 |
| Deep Learning | PyTorch + CUDA |
| Computer Vision | OpenCV |
| GUI | PyQt6 |
| GPU Monitoring | psutil + nvidia-smi |
| Optimization | TensorRT (optional) |

## 📄 License

MIT License — Feel free to use and modify!
