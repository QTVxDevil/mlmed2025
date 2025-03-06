import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import HC_Dataset, transform
from src.model import UNet
from src.config import TRAINING_DIR, TRAIN_CSV, BATCH_SIZE, EPOCHS, MODEL_DIR, LR

train_dataset = HC_Dataset(csv_file=TRAIN_CSV,
                           img_dir=TRAINING_DIR,
                           transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNet()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', unit='batch')
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss/len(train_loader))

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "unet_hc.pth"))
