import os

BASE_DIR = "dataset"
TRAINING_DIR = os.path.join(BASE_DIR, "training_set")
TEST_DIR = os.path.join(BASE_DIR, "test_set")
TRAIN_CSV = os.path.join(BASE_DIR, "training_set_pixel_size_and_HC.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_set_pixel_size.csv")

IMG_SIZE = (512, 512)
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0001
MODEL_DIR = "trained_model"
