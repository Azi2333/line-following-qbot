import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def image_find_objects_weighted_centroid(image, min_val=100, max_val=255, minArea=20, maxArea=50000):
    mask = cv2.inRange(image, min_val, max_val)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)

    weighted_sum = 0
    total_area = 0

    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if minArea < area < maxArea:
            weighted_sum += centroids[idx][0] * area
            total_area += area

    if total_area > 0:
        col = weighted_sum / total_area
        return col, total_area
    else:
        return None, 0

def process_folder(folder_path, max_error=640):
    error_list = []
    file_list = sorted(os.listdir(folder_path))
    print(f"Found {len(file_list)} images in {folder_path}")

    for filename in file_list:
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to read {filename}")
            continue

        center_x = image.shape[1] // 2
        col, total_area = image_find_objects_weighted_centroid(image,
                                                               min_val=100,
                                                               max_val=255,
                                                               minArea=20,
                                                               maxArea=50000)
        error = (col - center_x) if col is not None else max_error
        error_list.append(error)

    return error_list

def plot_error_lists(error_list_a, error_list_b, label_a="Dataset A", label_b="Dataset B", error_threshold=40):
    plt.figure(figsize=(12, 6))
    plt.plot(error_list_a, label=f"{label_a} Error", color="blue")
    plt.plot(error_list_b, label=f"{label_b} Error", color="red")
    # plt.axhline(y=error_threshold, color="green", linestyle="--", label=f"+{error_threshold} Threshold")
    # plt.axhline(y=-error_threshold, color="green", linestyle="--", label=f"-{error_threshold} Threshold")
    plt.xlabel("Frame Index")
    plt.ylabel("Offset from Center (pixels)")
    plt.title("Line Following Deviation Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------- 主程序 --------
folder_a = "classic_images"
folder_b = "cnn_images"  # <-- 你的另一个文件夹名，按需修改

error_list_a = process_folder(folder_a)
error_list_b = process_folder(folder_b)

plot_error_lists(error_list_a, error_list_b, label_a="Classic", label_b="CNN")
