import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from src.config import TRAINING_DIR, TEST_DIR, TRAINING_CSV, TEST_CSV, IMG_SIZE, BATCH_SIZE

class FetalHeadDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None, is_train=True):
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filename = self.data.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_train:
            mask_path = img_path.replace(".png", "_Annotation.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = self.transform(mask)
            mask = (mask > 0).to(torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return image, mask
        else:
            return image

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

train_dataset = FetalHeadDataset(TRAINING_DIR, TRAINING_CSV, transform=transform, is_train=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = FetalHeadDataset(TEST_DIR, TEST_CSV, transform=transform, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)