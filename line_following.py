# -----------------------------------------------------------------------------#
# ------------------Skills Progression 1 - Task Automation---------------------#
# -----------------------------------------------------------------------------#
# ----------------------------Lab 3 - Line Following---------------------------#
# -----------------------------------------------------------------------------#

# Imports
import os
from pal.products.qbot_platform import QBotPlatformDriver, Keyboard, \
    QBotPlatformCSICamera, QBotPlatformRealSense, QBotPlatformLidar
from qbot_platform_functions import QBPVision
from quanser.hardware import HILError
from pal.utilities.probe import Probe
from pal.utilities.gamepad import LogitechF710
import time
import numpy as np
import cv2
from qlabs_setup import setup

###################################################################################################################

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define SimpleCNN structure
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Calculate the flattened feature size after pooling (640x480 -> 80x60 after three poolings)
        self.fc1 = torch.nn.Linear(64 * 80 * 60, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 640x480 -> 320x240
        x = self.pool(torch.relu(self.conv2(x)))  # 320x240 -> 160x120
        x = self.pool(torch.relu(self.conv3(x)))  # 160x120 -> 80x60
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ========== 2. Load Trained Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = "dataset"  # Windows
num_classes = len(os.listdir(dataset_path))

model = SimpleCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(r"cnn_train\best_cnn_model_640x480.pth", map_location=device))
model.eval()  # Set to inference mode

print("✅ Model weights successfully loaded!")

# ========== 3. Define Image Preprocessing ==========
transform = transforms.Compose([
    transforms.Resize((640, 480)),  # Ensure correct input size
    transforms.ToTensor()
])

# ========== 4. Load and Predict Test Image ==========

class_names = ["No line detected",
               "Straight - Centered", "Straight - Slightly Right (Adjust Right)",
               "Straight - Slightly Left (Adjust Left)",
               "Right Turn - Centered", "Right Turn - Slightly Right (Adjust Right)",
               "Right Turn - Slightly Left (Adjust Left)"]  # Modify based on your class names

def predict_image(image):
    if isinstance(image, np.ndarray):  # Convert numpy array to PIL.Image if needed
        image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)  # Get predicted class index

    class_names = sorted(os.listdir(dataset_path))  # Get class names
    predicted_label = class_names[predicted_class.item()]

    print(f"🔍 Predicted Class: {predicted_label} ({predicted_class.item()})")
    return class_names[predicted_class.item()]


def map_prediction_to_speed(pred):
    """
    Map the predicted class to (forward speed, turn speed).
    Adjust these values as needed.
    """
    if pred in ["Straight - Slightly Right (Adjust Right)"]:
        return 0.1, -0.1
    elif pred == "Right Turn - Slightly Right (Adjust Right)":
        return 0.1, -0.25
    elif pred == "Straight - Centered":
        return 0.1, 0.0
    elif pred in ["Straight - Slightly Left (Adjust Left)", "Right Turn - Slightly Left (Adjust Left)"]:
        return 0.1, 0.1
    elif pred == "Right Turn - Centered":
        return 0.1, -0.15
    elif pred == "No line detected":
        return -0.1, 0.0
    else:
        return 0.0, 0.0

##########################################################################################

# Section A - Setup
frame_count = 10000
setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)
ipHost, ipDriver = 'localhost', 'localhost'
commands, arm, noKill = np.zeros((2), dtype=np.float64), 0, True
frameRate, sampleRate = 60.0, 1 / 60.0
counter, counterDown = 0, 0
endFlag, offset, forSpd, turnSpd = False, 0, 0, 0
startTime = time.time()

save_dir = "cnn_images"
os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在就创建

def elapsed_time():
    return time.time() - startTime

timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017

try:
    # Section B - Initialization
    myQBot = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam = QBotPlatformCSICamera(frameRate=frameRate, exposure=39.0, gain=17.0)
    keyboard = Keyboard()
    vision = QBPVision()
    probe = Probe(ip=ipHost)
    probe.add_display(imageSize=[480, 640, 1], scaling=True, scalingFactor=2, name='Raw Image')
    probe.add_display(imageSize=[480, 640, 1], scaling=False, scalingFactor=2, name='Binary Image')
    line2SpdMap = vision.line_to_speed_map(sampleRate=sampleRate, saturation=75)
    next(line2SpdMap)
    startTime = time.time()
    time.sleep(0.5)

    # Main loop
    while noKill and not endFlag:
        t = elapsed_time()
        if not probe.connected:
            probe.check_connection()
        if probe.connected:
            newkeyboard = keyboard.read()
            if newkeyboard:
                arm = keyboard.k_space
                lineFollow = keyboard.k_7
                keyboardComand = keyboard.bodyCmd
                if keyboard.k_u:
                    noKill = False
            if not lineFollow:
                commands = np.array([keyboardComand[0], keyboardComand[1]], dtype=np.float64)
            else:
                commands = np.array([forSpd, turnSpd], dtype=np.float64)
            newHIL = myQBot.read_write_std(timestamp=time.time() - startTime, arm=arm, commands=commands)
            if newHIL:
                newDownCam = downCam.read()
                if newDownCam:
                    counterDown += 1
                    undistorted = vision.df_camera_undistort(downCam.imageData)
                    gray_sm = cv2.resize(undistorted, (640, 480))

                    img_name = f"frame_{counterDown:04d}.png"
                    img_path = os.path.join(save_dir, img_name)
                    cv2.imwrite(img_path, gray_sm)

                    _, binary = cv2.threshold(gray_sm, 128, 255, cv2.THRESH_BINARY)
                    result = predict_image(binary)
                    forSpd, turnSpd = map_prediction_to_speed(result)
except KeyboardInterrupt:
    print('User interrupted.')
finally:
    downCam.terminate()
    myQBot.terminate()
    probe.terminate()
    keyboard.terminate()


