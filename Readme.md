# 🦯 Object Detection and Audio Assistance System for Visually Impaired Persons

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv" />
  <img src="https://img.shields.io/badge/YOLOv3-Darknet-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/TTS-pyttsx3-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-Academic-lightgrey?style=for-the-badge" />
</p>

> A real-time, fully **offline** assistive system that detects objects in a live camera feed and announces them — along with their estimated distance — via audio feedback. Designed to enhance independence and safety for visually impaired individuals.

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration & Calibration](#configuration--calibration)
- [Usage](#usage)
- [Results](#results)
- [Performance Metrics](#performance-metrics)
- [Known Limitations](#known-limitations)
- [Future Scope](#future-scope)
- [Team](#team)
- [Acknowledgements](#acknowledgements)
- [References](#references)

---

## 🔍 About the Project

This project was developed as a **Final Year B.E. (CSE) project** at **Nitte Meenakshi Institute of Technology, Bengaluru** under **Visvesvaraya Technological University (VTU), Belagavi** for the academic year **2025–26**.

According to the WHO, over **2.2 billion people** worldwide experience some form of visual impairment. Conventional aids like white canes and guide dogs are limited — they can detect nearby barriers but cannot identify *what* those objects are or *how far* they are.

This system bridges that gap by combining:
- **Computer Vision** (Haar Cascade + YOLOv3) for real-time object detection
- **Monocular Distance Estimation** using the pin-hole camera model
- **Text-to-Speech (TTS)** via `pyttsx3` for instant, offline audio feedback

The entire pipeline runs **without any internet connection**, making it practical for everyday use in any environment.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎯 **Multi-Object Detection** | Detects 80+ object categories using YOLOv3 (COCO-trained) |
| 📏 **Distance Estimation** | Estimates object distance in cm using the pin-hole camera model |
| 🔊 **Audio Feedback** | Converts detected object + distance to speech via `pyttsx3` |
| 📡 **Fully Offline** | No internet required — all processing done on-device |
| ⚡ **Real-Time Processing** | Operates at 15–25 FPS on standard hardware |
| 👤 **Face Detection** | Fast face/person detection using Haar Cascade Classifier |
| 💰 **Low-Cost Hardware** | Runs on a standard laptop; deployable on Raspberry Pi |
| 🔕 **Smart Audio Throttling** | Prevents repetitive announcements with a delay mechanism |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│              Live Camera Feed (Webcam / Pi Cam)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  COMPUTER VISION MODULE                     │
│   ┌─────────────────────┐   ┌─────────────────────────┐    │
│   │  Haar Cascade       │   │       YOLOv3             │    │
│   │  (Fast Face/Person  │   │  (80+ Object Classes,   │    │
│   │   Detection)        │   │   Multi-Scale Detection) │    │
│   └─────────────────────┘   └─────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────┘
                           │ Bounding Boxes + Labels
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                DISTANCE ESTIMATION MODULE                   │
│       Formula: Distance = (Real Width × Focal Length)       │
│                           / Perceived Width                 │
│       (Calibrated using reference image at known distance)  │
└──────────────────────────┬──────────────────────────────────┘
                           │ Object Name + Distance (cm)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  SPEECH OUTPUT MODULE                       │
│             pyttsx3 TTS Engine (Fully Offline)              │
│         "Person detected at 129 centimetres"                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| Object Detection | YOLOv3 (Darknet weights via OpenCV DNN) |
| Face Detection | Haar Cascade Classifier (OpenCV) |
| Image Processing | OpenCV (`cv2`) |
| Numerical Computing | NumPy |
| Text-to-Speech | `pyttsx3` |
| Distance Estimation | Pin-hole camera model (monocular) |

---

## ✅ Prerequisites

- Python **3.8 or higher**
- A working **webcam** (built-in or USB)
- ~250 MB free disk space (for YOLOv3 weights)
- Operating System: Windows / Linux / macOS

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/object-detection-audio-assistance.git
cd object-detection-audio-assistance
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
opencv-python>=4.5.0
numpy>=1.21.0
pyttsx3>=2.90
```

### 4. Download YOLOv3 Weights & Config

YOLOv3 weights are too large to include in this repo. Download them manually:

```bash
# Create the models directory
mkdir -p models/yolov3

# Download YOLOv3 weights (~236 MB)
wget https://pjreddie.com/media/files/yolov3.weights -P models/yolov3/

# Download config and class names
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -P models/yolov3/
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -P models/yolov3/
```

> **Windows users:** Download the files manually from the URLs above and place them inside `models/yolov3/`.

### 5. Linux-only: Fix pyttsx3 Audio

```bash
sudo apt-get install espeak ffmpeg libespeak1
```

---

## 📁 Project Structure

```
object-detection-audio-assistance/
│
├── models/
│   ├── yolov3/
│   │   ├── yolov3.weights          # YOLOv3 pre-trained weights (download separately)
│   │   ├── yolov3.cfg              # YOLOv3 network configuration
│   │   └── coco.names              # 80 COCO class labels
│   └── haar/
│       └── haarcascade_frontalface_default.xml   # OpenCV Haar Cascade (bundled with OpenCV)
│
├── reference/
│   └── ref_image.jpg               # Reference image for focal length calibration
│
├── src/
│   ├── main.py                     # Entry point — runs the full detection pipeline
│   ├── detector.py                 # YOLO + Haar Cascade detection logic
│   ├── distance.py                 # Focal length calibration & distance estimation
│   ├── tts_engine.py               # Text-to-speech audio feedback module
│   └── utils.py                    # Helper functions (NMS, drawing bounding boxes, etc.)
│
├── results/
│   └── screenshots/                # Sample output frames (from testing)
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Configuration & Calibration

### Calibration (Important — Do This First!)

The distance estimation relies on knowing your camera's **focal length**, computed once using a reference image.

1. Place a person or known object **exactly 100 cm (1 metre)** from your camera.
2. Take a photo and save it as `reference/ref_image.jpg`.
3. Measure the known real-world width of the object (e.g., average shoulder width ≈ 45 cm).
4. Update these constants in `src/distance.py`:

```python
KNOWN_DISTANCE = 100.0      # Distance in cm at which reference image was taken
KNOWN_WIDTH    = 45.0       # Real-world width of the reference object in cm
```

The system will auto-compute the focal length at startup using the formula:

```
Focal Length (f) = (Pixel Width × Known Distance) / Known Width
```

### Confidence & NMS Thresholds

In `src/detector.py`:

```python
CONFIDENCE_THRESHOLD = 0.5   # Minimum confidence to consider a detection
NMS_THRESHOLD        = 0.4   # Non-Maximum Suppression overlap threshold
INPUT_SIZE           = 416   # YOLO input resolution (416×416)
```

### Audio Announcement Cooldown

In `src/tts_engine.py`:

```python
ANNOUNCEMENT_COOLDOWN = 3    # Seconds between repeated announcements of the same object
```

---

## ▶️ Usage

### Run the Main Application

```bash
python src/main.py
```

### Controls

| Key | Action |
|---|---|
| `q` | Quit the application |
| `s` | Save current frame as screenshot |
| `m` | Toggle audio on/off |

### What You'll See

- A live camera window titled **"Object Distance Detection"**
- Each detected object is enclosed in a **coloured bounding box**
- The object **label and estimated distance (cm)** are displayed above the box
- Your speakers will announce, e.g.: *"Person, 129 centimetres"*

---

## 📊 Results

The system was evaluated under normal indoor lighting using a laptop webcam. Results across test scenarios:

| Test Scenario | Detected Objects | Estimated Distances | Actual Distance |
|---|---|---|---|
| Person + Mobile Phone | ✅ Person, ✅ Cell Phone | 129 cm, 84 cm | ~130 cm, ~85 cm |
| Person + Book | ✅ Person, ✅ Book | 111 cm, 95 cm | ~110 cm, ~95 cm |
| Person + Cup | ✅ Person, ✅ Cup | 107 cm, 102 cm | ~105 cm, ~100 cm |
| Person only (far) | ✅ Person | 136 cm | ~135 cm |
| Person only (near) | ✅ Person | 126 cm | ~125 cm |

> Distance deviations of ±5–10 cm are within the expected margin for monocular estimation and are acceptable for assistive navigation purposes.

---

## 📈 Performance Metrics

Tested on a standard laptop (Intel Core i5, 8GB RAM, no dedicated GPU):

| Metric | Value |
|---|---|
| Average FPS | ~7.83 FPS (CPU-only) |
| CPU Usage | ~84.3% |
| RAM Usage | ~35.5% |
| Detection Latency | < 130 ms per frame |
| TTS Response Time | < 200 ms |
| Similarity Score (DrillBit) | 11% |

> On GPU-enabled hardware or a Jetson Nano, FPS is expected to reach 25–30.

---

## ⚠️ Known Limitations

- **Lighting Sensitivity:** Haar Cascade performance degrades under poor or uneven lighting.
- **Monocular Estimation Variance:** Distance accuracy varies with changes in object orientation or partial occlusion.
- **CPU Bottleneck:** YOLOv3 on CPU-only hardware yields ~7–8 FPS; GPU acceleration is recommended for smoother experience.
- **Fixed Object Widths:** Distance estimation assumes fixed real-world widths per class; accuracy varies across individuals and object sizes.
- **No Outdoor GPS Navigation:** The system does not include location-based or GPS-assisted outdoor guidance.
- **Controlled Environment Testing:** Full validation with visually impaired users in outdoor environments has not yet been conducted.

---

## 🔮 Future Scope

- [ ] **Directional Audio Cues** — Inform users whether an object is to their left, right, or centre
- [ ] **Stereo Depth Estimation** — Use dual cameras for more accurate distance measurement
- [ ] **Raspberry Pi / Jetson Nano Deployment** — Fully portable, wearable form factor
- [ ] **Voice Command Interface** — Allow users to ask "what's in front of me?" verbally
- [ ] **Outdoor Navigation Support** — GPS integration for street-level guidance
- [ ] **YOLOv8 / Faster RCNN Upgrade** — Improved accuracy and speed
- [ ] **Real-World User Testing** — Formal usability study with visually impaired participants
- [ ] **Priority-Based Audio Alerts** — Announce closest/most dangerous objects first
- [ ] **OCR Integration** — Read text from signs, labels, or screens aloud

---

## 👥 Team

This project was developed by final-year B.E. (CSE) students at **Nitte Meenakshi Institute of Technology, Bengaluru**:

| Name | USN | Contact |
|---|---|---|
| Aayush Thapa | 1NT22CS003 | aayushapa640@gmail.com |
| Abhay Bhagat | 1NT22CS005 | — |
| Abhishek Kumar Raut | 1NT22CS010 | hek899@gmail.com |
| Prashant Adhikari | 1NT22CS136 | prashant.kyler99@gmail.com |

**Project Guide:** Dr. Sreenivasa N, Professor, Dept. of CSE, NMIT Bengaluru
**HoD:** Dr. Vijaya Shetty S
**Principal:** Dr. H. C. Nagaraj

---

## 🙏 Acknowledgements

We extend our heartfelt gratitude to:
- **Dr. Sreenivasa N** — Project Guide, for his constant mentorship and feedback
- **Dr. Vijaya Shetty S** — Head of Department, CSE, NMIT
- **Dr. H. C. Nagaraj** — Principal, NMIT Bengaluru
- The **OpenCV** and **Darknet/YOLO** open-source communities
- **Visvesvaraya Technological University (VTU), Belagavi**

---

## 📚 References

1. P. Viola and M. Jones, "Rapid Object Detection using a Boosted Cascade of Simple Features," IEEE CVPR, 2001.
2. J. Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection," IEEE CVPR, 2016.
3. J. Redmon and A. Farhadi, "YOLOv3: An Incremental Improvement," arXiv, 2018.
4. A. Abdurrazyid, "Face Detection and Global Positioning System on a Tool for Blind People," EEI, Vol. 11, Issue 3, 2022.
5. "VISIONSPEAK: Object Detection and Voice Assistance for Visually Impaired People," IJARCCE, 2025.
6. R. B. Islam et al., "Deep learning based object detection and surrounding environment description for visually impaired people," Heliyon, 2023.
7. F. Yao, W. Zhou and H. Hu, "A Review of Vision-Based Assistive Systems for Visually Impaired People," arXiv, May 2025.

---

## 📄 License

This project is submitted for academic purposes under **Visvesvaraya Technological University**. All rights reserved by the authors. For any reuse or extension, please contact the team.

---

<p align="center">
  Made with ❤️ at Nitte Meenakshi Institute of Technology, Bengaluru — 2025–26
</p>
