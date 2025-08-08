import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset


# Функція зміни масштабу
def resize_with_aspect_ratio(image, target_height=360):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)
    return cv2.resize(image, (target_width, target_height))


class TUSimpleDataset(Dataset):
    def __init__(self, data_path, json_mapping, folders, target_height=360):
        self.data_path = data_path
        self.json_mapping = json_mapping
        self.folders = folders
        self.target_height = target_height
        self.data = []

        for folder in self.folders:
            json_file = self.json_mapping[folder]
            json_path = os.path.join(self.data_path, json_file)

            with open(json_path, 'r') as f:
                for line in f:
                    label = json.loads(line.strip())
                    if label["raw_file"].startswith(f"clips/{folder}"):
                        self.data.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx]
        img_path = os.path.join(self.data_path, label["raw_file"])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img / 255.0
        mask = np.zeros((720, 1280), dtype=np.uint8)
        lanes = label["lanes"]
        h_samples = label["h_samples"]

        for lane in lanes:
            points = [(int(x), int(y)) for x, y in zip(lane, h_samples) if x >= 0]
            if points:
                cv2.polylines(mask, [np.array(points)], isClosed=False, color=1, thickness=20)

        img = resize_with_aspect_ratio(img, target_height=self.target_height)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask
