import segmentation_models_pytorch as smp
from src.config import device

def get_model():
    model = smp.DeepLabV3(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    return model.to(device)