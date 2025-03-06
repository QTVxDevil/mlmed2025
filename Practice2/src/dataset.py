import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class HC_Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        mask_name = img_name.replace(".png", "_Annotation.png")
        
        image = Image.open(img_name).convert("L")
        mask = Image.open(mask_name).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = HC_Dataset(csv_file="dataset/training_set_pixel_size_and_HC.csv",
                           img_dir="dataset/training_set",
                           transform=transforms)