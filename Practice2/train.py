import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import train_loader
from src.model import get_model
from src.config import MODEL_DIR, EPOCHS, LR, device

model = get_model()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.squeeze(1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "deeplabv3.pth"))