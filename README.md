# 🔧 Classic Line-Following Algorithm for QBot

This is the implementation of a **classical control-based line-following system** for the Quanser QBot, developed as part of a university group project for the Advanced AI module.

---

## 📌 Overview

This module uses a **bottom camera**, image thresholding, and simple **centroid detection + kinematic control** to enable QBot to follow lines in a simulated warehouse environment.

The project was implemented in Python using the Quanser API, OpenCV, and a control mapping based on heading angle deviation.

---

## 🧠 Key Features

- Grayscale image capture using QBot's bottom camera
- Image slicing + thresholding to detect the line area
- Object detection via connected components to locate the line centroid
- Error computation between centroid and image center
- Forward & angular velocity mapping using basic trig and differential drive kinematics
- Real-time visual feedback via Probe
- Optional frame saving for later training (CNN)

---

## 🗂 Files in this Branch

| File | Description |
|------|-------------|
| `line_following.py` | Main logic loop for classical control navigation |
| `qbot_platform_functions.py` | Custom helper functions for image processing & movement control |
| `output.png` | Sample output visualization |
| `Individual_Project_Template.ipynb` | Full explanation and evaluation (can export as PDF for sharing/reporting) |

---

## 🚀 How to Run

1. Launch the Quanser QBot Warehouse Simulation
2. Run `line_following.py` Run `observer.py`
3. Press keyboard keys to arm/start the robot, and observe line following behavior
4. Optional: capture binary images for CNN training by enabling `save_data`
