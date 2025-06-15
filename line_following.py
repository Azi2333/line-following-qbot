import os
import cv2
import numpy as np
import time
from pal.products.qbot_platform import QBotPlatformDriver, Keyboard, \
    QBotPlatformCSICamera, QBotPlatformRealSense, QBotPlatformLidar
from qbot_platform_functions import QBPVision
from quanser.hardware import HILError
from pal.utilities.probe import Probe
from pal.utilities.gamepad import LogitechF710
from qlabs_setup import setup

# 创建数据保存函数
def save_data(image, output_dir, frame_id):
    """
    保存二值化图像
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存二值化图像
    image_filename = os.path.join(output_dir, f"image_{frame_id:04d}.png")
    cv2.imwrite(image_filename, image)

# 设备初始化
setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)
ipHost, ipDriver = 'localhost', 'localhost'
commands, arm, noKill = np.zeros((2), dtype=np.float64), 0, True
frameRate, sampleRate = 60.0, 1 / 60.0
counter, counterDown = 0, 0
endFlag, offset, forSpd, turnSpd = False, 0, 0, 0
startTime = time.time()

def elapsed_time():
    return time.time() - startTime

timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017

try:
    # 设备初始化
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

    # 运行循环
    while noKill and not endFlag:
        t = elapsed_time()

        if not probe.connected:
            probe.check_connection()

        if probe.connected:
            # 读取键盘输入
            newkeyboard = keyboard.read()
            if newkeyboard:
                arm = keyboard.k_space
                lineFollow = keyboard.k_7
                keyboardComand = keyboard.bodyCmd
                if keyboard.k_u:
                    noKill = False

            # 控制机器人运动
            if not lineFollow:
                commands = np.array([keyboardComand[0], keyboardComand[1]], dtype=np.float64)
            else:
                commands = np.array([forSpd, turnSpd], dtype=np.float64)

            # 读取 QBot 传感器数据
            newHIL = myQBot.read_write_std(timestamp=time.time() - startTime, arm=arm, commands=commands)
            if newHIL:
                timeHIL = time.time()
                newDownCam = downCam.read()
                if newDownCam:
                    counterDown += 1

                    # 处理图像
                    undistorted = vision.df_camera_undistort(downCam.imageData)
                    gray_sm = cv2.resize(undistorted, (320, 200))
                    gray_sm2 = cv2.resize(undistorted, (640, 480))
                    save_data(gray_sm2, "test_project", counterDown)

                    # 进行二值化处理
                    binary = vision.subselect_and_threshold(gray_sm, 180, 255, 50, 100)

                    # 轨迹检测
                    col, row, area = vision.image_find_objects(binary, connectivity=8, minArea=500, maxArea=4000)

                    # 计算速度控制
                    forSpd, turnSpd = line2SpdMap.send((col, 1, 70))

                    # **保存二值化图像**
           

                if counterDown % 4 == 0:
                    probe.send(name='Raw Image', imageData=gray_sm)
                    probe.send(name='Binary Image', imageData=binary)

                prevTimeHIL = timeHIL

except KeyboardInterrupt:
    print('User interrupted.')
except HILError as h:
    print(h.get_error_message())
finally:
    downCam.terminate()
    myQBot.terminate()
    probe.terminate()
    keyboard.terminate()
