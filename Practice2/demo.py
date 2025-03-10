import os
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from src.model import get_model
from src.config import device, IMG_SIZE, TEST_DIR

model = get_model()
model.load_state_dict(torch.load("trained_model/deeplabv3.pth", map_location=device))
model.eval()

def predict(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    input_tensor = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
    output = torch.sigmoid(output).cpu().numpy().squeeze()

    mask = (output > 0.5).astype(np.uint8) * 255
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    mask_resized = cv2.resize(mask_colored, (image.shape[1], image.shape[0]))

    blended = cv2.addWeighted(image, 0.7, mask_resized, 0.3, 0)

    return blended

if __name__ == "__main__":
    test_image_path = os.path.join(TEST_DIR, "001_HC.png")
    result = predict(test_image_path)
    cv2.imwrite("demo.png", result)
