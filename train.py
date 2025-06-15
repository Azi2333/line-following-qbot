import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt  # ✅ 用于绘图


# ======================== 1. 自定义数据集类 ========================
class CustomDataset(Dataset):
    def __init__(self, root_dir, mode="train", transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.data = []
        self.classes = sorted(os.listdir(root_dir))
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')  # 允许的图片格式
        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name, mode)
            if not os.path.isdir(class_path):
                continue
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_path.lower().endswith(valid_extensions):
                    self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('L')  # 转换为灰度图 (单通道)
        if self.transform:
            image = self.transform(image)
        return image, label


# ======================== 2. CNN 模型 (支持 640x480) ========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 计算展平后特征图的大小 (640x480 -> 80x60 经过三次池化)
        self.fc1 = nn.Linear(64 * 80 * 60, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 640x480 -> 320x240
        x = self.pool(torch.relu(self.conv2(x)))  # 320x240 -> 160x120
        x = self.pool(torch.relu(self.conv3(x)))  # 160x120 -> 80x60
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ======================== 3. 训练 & 测试循环 ========================
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# ======================== 4. 训练代码 (使用 640x480) ========================
if __name__ == "__main__":
    device = torch.device("cuda:0")
    best_accuracy = 0.0
    best_model_path = "best_cnn_model_640x480.pth"
    dataset_path = r"C:\Users\qwer\Documents\Quanser\Quanser_Tutorial_Exercises-20250226\Quanser Tutorial Exercises for Rankine509\Moodle_Quanser_Files\sp1_task_automation\l3_line_following\digital_twin\python\prefilter\dataset_group7"  # Windows
    # dataset_path = "/home/user/dataset"  # Linux

    transform = transforms.Compose([
        transforms.Resize((640, 480)),  # ✅ 直接使用 640x480
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(root_dir=dataset_path, mode="train", transform=transform)
    val_dataset = CustomDataset(root_dir=dataset_path, mode="val", transform=transform)
    test_dataset = CustomDataset(root_dir=dataset_path, mode="test", transform=transform)

    batch_size = 16  # 由于图像较大，减少 batch size 避免显存溢出
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(os.listdir(dataset_path))
    model = SimpleCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_accuracy = test(model, val_loader, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"🎯 更优模型已保存到 {best_model_path}")

    test_accuracy = test(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    sys.exit()
