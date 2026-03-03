# 🧠 NeuroDrive Perception Lab

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20OpenCV-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active%20R%26D-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Autonomous%20Driving%20%7C%20ADAS-red)

> **A modular R&D framework for autonomous driving perception, focusing on hybrid architectures, temporal consistency, and real-time edge optimization.**

---

## 🚀 Overview

**NeuroDrive Perception Lab** is an experimental playground designed to bridge the gap between academic computer vision papers and robust automotive applications. Unlike standard implementations that rely solely on deep learning or classic vision, this repository explores **Hybrid Perception Architectures**.

The core philosophy is to combine the **determinism of geometric computer vision** with the **semantic understanding of Deep Learning** to achieve higher reliability in challenging scenarios (e.g., occlusion, harsh weather, curved roads).

## ⚡ Key Features & Innovations

### 1. Hybrid Lane Analysis (Spatial-Geometric)
Instead of relying purely on segmentation masks (which can be noisy), this module implements a fused pipeline:
*   **Deep Semantic Branch:** Uses a lightweight segmentation network to identify road regions.
*   **Geometric Branch:** Applies adaptive thresholding and Inverse Perspective Mapping (IPM).
*   **Innovation:** A **Confidence-Weighted Fusion** algorithm that dynamically prioritizes the geometric branch on clear highways and the deep branch on urban roads.

### 2. Spatio-Temporal Object Tracking
Detection alone is insufficient for decision making. This module implements:
*   **Motion Modeling:** Extended Kalman Filter (EKF) for state estimation.
*   **Data Association:** IoU-based matching coupled with visual appearance embeddings (ReID).
*   **Innovation:** **"Occlusion Recovery Logic"** that maintains object trajectory memory using LSTM cells when visual contact is temporarily lost.

### 3. Modular Architecture
Designed for scalability. You can swap the backend detector (e.g., YOLOv8 to RT-DETR) or the tracking logic (SORT to ByteTrack) without breaking the rest of the pipeline.

## 📂 Project Structure

```text
NeuroDrive-Perception-Lab/
├── assets/                 # Demo GIFs and images
├── configs/                # YAML configuration files for models
├── data/                   # Input samples (Ignored by Git)
├── notebooks/              # Prototyping & Visualization (Jupyter)
├── src/                    # Main Source Code
│   ├── core/               # Math kernels, Geometry, Kalman Filters
│   ├── modules/            # Perception Algorithms
│   │   ├── lanes/          # Hybrid Lane Detection Logic
│   │   ├── objects/        # Object Detection & Tracking wrappers
│   │   └── fusion/         # Sensor Fusion Logic
│   └── utils/              # Visualization & I/O helpers
├── tests/                  # Unit tests
└── requirements.txt        # Dependencies
```

## 🛠️ Getting Started

### Prerequisites
All modules are designed to run in **Google Colab** or a local Python environment.

```bash
# Clone the repository
git clone https://github.com/YourUsername/NeuroDrive-Perception-Lab.git
cd NeuroDrive-Perception-Lab

# Install dependencies
pip install -r requirements.txt
```

### Running the Modules
Each module can be tested via the provided notebooks or command-line scripts.

**Example: Hybrid Lane Detection**
```bash
python -m src.modules.lanes.hybrid_demo --input data/road_sample.mp4 --debug
```

## 📊 Roadmap & Research
- [ ] **Phase 1:** Geometric Lane Detection (Sliding Window + IPM).
- [ ] **Phase 2:** Hybrid Fusion (Classic CV + Semantic Segmentation).
- [ ] **Phase 3:** Multi-Object Tracking with Kalman Filter & SORT.
- [ ] **Phase 4:** Real-time optimization using TensorRT for NVIDIA Edge devices.

## 🤝 Contribution
This is an open research project. Issues and Pull Requests regarding optimization, new architectures, or edge cases are welcome.

---
*Developed by kazem sahebi - Computer Vision & ADAS Engineer.*
