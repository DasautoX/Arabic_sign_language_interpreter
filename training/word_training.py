import os
import cv2
import numpy as np
import threading
import queue
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression as LR
import mediapipe as mp
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import DatasetFolder
from torchvision import transforms

# Mediapipe setup
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

pose_model_path = 'assets/word_pose_landmarker_lite.task'
hand_model_path = 'assets/word_hand_landmarker.task'


pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.VIDEO)

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2)

def mediapipe_detection(image, frame_timestamp_ms, poselandmarker, handlandmarker):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.flip(image_rgb, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    pose_results = poselandmarker.detect_for_video(mp_image, frame_timestamp_ms)
    hand_results = handlandmarker.detect_for_video(mp_image, frame_timestamp_ms)

    return pose_results, hand_results

def extract_keypoints(pose_results, hand_results):
    hands = hand_results.hand_world_landmarks
    hands_npy = np.zeros(21*2*2)
    i = 0
    for hand in hands:
        for landmark in hand:
            hands_npy[i:i+2] = [landmark.x, landmark.y]
            i += 2

    bodies = pose_results.pose_world_landmarks
    body_npy = np.zeros(33*2)
    i = 0
    for body in bodies:
        for landmark in body:
            body_npy[i:i+2] = [landmark.x, landmark.y]
            i += 2

    landmarks = np.concatenate([hands_npy, body_npy], axis=0)
    return landmarks

def video_to_npy(video_path):
    cap = cv2.VideoCapture(video_path)
    video_data = []
    with PoseLandmarker.create_from_options(pose_options) as poselandmarker:
        with HandLandmarker.create_from_options(hand_options) as handlandmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                landmarks = extract_keypoints(*mediapipe_detection(frame, frame_timestamp_ms, poselandmarker, handlandmarker))
                video_data.append(landmarks)
            cap.release()

    return np.array(video_data)

# Data cleaning

def estimate_nan(x, y, x_missed):
    x = np.asarray(x).reshape(-1, 1)
    x_missed = np.asarray(x_missed).reshape(-1, 1)
    poly = PolynomialFeatures(7)
    x_poly = poly.fit_transform(x)
    x_missed_poly = poly.transform(x_missed)
    lr = LR()
    lr.fit(x_poly, y)
    
    return lr.score(x_poly, y), lr.predict(x_missed_poly)

def fill_missing_values(arr):
    def find_valid_range(vec):
        start, end = 0, len(vec)
        for i in range(len(vec)):
            if vec[i] != 0:
                start = i
                break
        for i in range(len(vec)-1, -1, -1):
            if vec[i] != 0:
                end = i + 1
                break
        return start, end

    for i in range(arr.shape[1]):
        col = arr[:, i]
        start, end = find_valid_range(col)
        if start < end:
            non_zero_values = col[start:end]
            nan_indices = np.where(col == 0)[0]
            if len(nan_indices) > 0:
                x_known = np.arange(len(non_zero_values))
                y_known = non_zero_values
                x_missing = np.arange(start, end)[col[start:end] == 0]
                _, predicted = estimate_nan(x_known, y_known, x_missing)
                col[x_missing] = predicted

# Model training

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, device='cpu'):
        super(GRUModel, self).__init__()
        self.layers = nn.ModuleList()
        self.hidden_states = []
        self.device = device
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(nn.GRU(input_size if i == 0 else hidden_sizes[i-1], hidden_size, batch_first=True))
            self.hidden_states.append(torch.zeros(1, 1, hidden_size, device=device))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x, self.hidden_states[i] = layer(x, self.hidden_states[i].detach())
        return self.output_layer(x[:, -1, :])

# Paths
raw_data_path = 'data/raw'
npy_data_path = 'data/npy'

# Video processing
for video_file in os.listdir(raw_data_path):
    video_npy = video_to_npy(os.path.join(raw_data_path, video_file))
    save_path = os.path.join(npy_data_path, os.path.splitext(video_file)[0] + '.npy')
    np.save(save_path, video_npy)

# Data cleaning
for npy_file in os.listdir(npy_data_path):
    npy_data = np.load(os.path.join(npy_data_path, npy_file))
    fill_missing_values(npy_data)
    np.save(os.path.join(npy_data_path, npy_file), npy_data)

# Dataset and training
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.astype(np.float32))
])

dataset = DatasetFolder(npy_data_path, np.load, extensions=['.npy'], transform=transform)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = GRUModel(input_size=134, hidden_sizes=[64, 32], output_size=len(dataset.classes), device='cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(10):
    model.train()
    for data, targets in train_loader:
        data, targets = data.cuda(), targets.cuda()
        outputs = model(data)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")