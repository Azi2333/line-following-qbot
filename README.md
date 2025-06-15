# 🤖 CNN-Based Line-Following Robot (Quanser QBot)

This repository implements a **Convolutional Neural Network (CNN)** solution to enable autonomous line-following behavior in a simulated warehouse environment using **Quanser QBot**. It is a deep-learning alternative to classical control logic.

---

## 📌 Project Overview

- Platform: Quanser Interactive Labs (QBot + Warehouse Simulation)
- Goal: Navigate warehouse floor by following a white line using camera vision
- Control Strategy: Deep learning model (CNN) classifies each frame into movement direction
- Language: Python (PyTorch + OpenCV)
- Deployment: Model inference in real-time via QBot camera + motor control

---

## 🧠 Neural Network Architecture

A lightweight CNN architecture designed for 640x480 grayscale images:

```python
Input (1x640x480)
→ Conv2d(1→16) + ReLU + MaxPool
→ Conv2d(16→32) + ReLU + MaxPool
→ Conv2d(32→64) + ReLU + MaxPool
→ Flatten → FC(512) → FC(num_classes)
```

- Trained with CrossEntropyLoss
- Optimizer: Adam (lr=1e-4)
- Epochs: 100
- Best model saved as: `best_cnn_model_640x480.pth`

---

## 📁 Dataset Structure

Each image is labeled into directional class (e.g. straight, left, right).

```text
dataset_group7/
├── center/
│   ├── train/
│   ├── val/
│   └── test/
├── left/
├── right/
```

Images are grayscale, resized to 640x480, and collected from real QBot runs.

> Note: Full dataset not included due to size. Please use your own or request sample.

---

## 🏁 Training the Model

```bash
python train.py
```

Inside `train.py`:
- Model is trained using `CustomDataset` loader
- Automatically evaluates on validation set each epoch
- Best model saved based on validation accuracy

---

## 🚀 Deployment: Real-Time Prediction on QBot

The file `line_following.py`:
- Loads `best_cnn_model_640x480.pth`
- Captures live bottom camera feed
- Applies transform + CNN prediction
- Maps prediction to movement command
- Commands QBot via differential drive

```bash
python line_following.py
```

---

## 📊 Evaluation: CNN vs Classical Control

`eva.py` runs a frame-wise offset comparison between classic and CNN controllers.

It measures:
- **Pixel deviation** from center of line
- Frame-by-frame performance
- Area under curve (tracking accuracy)

Output:

![Evaluation Result](evaluation_result_with_label.svg)

---

## 🧰 File Guide

| File | Purpose |
|------|---------|
| `train.py` | CNN model training |
| `line_following.py` | Real-time control using trained model |
| `qlabs_setup.py` | Simulation setup (QBot + environment) |
| `qbot_platform_functions.py` | Vision and movement logic |
| `eva.py` | Evaluation plot generation |
| `observer.py` | Image visualizer |
| `best_cnn_model_640x480.pth` | Trained CNN model |
| `evaluation_result_with_label.svg` | Evaluation plots |

---

## 🧠 What You’ll Learn

- How to train CNNs for image-based classification tasks
- How to integrate model inference into real-time robotics
- How to benchmark learning-based methods against control logic
- How to simulate robots using Quanser’s QLabs API

---

## 📦 Requirements

```bash
pip install torch torchvision opencv-python matplotlib pillow
```

> Note: Requires Quanser QLabs software + QBot simulation environment.

---

