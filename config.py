import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR_A = "./data/train/winter"
TRAIN_DIR_B = "./data/train/summer"
VAL_DIR = "./data/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 1.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 150
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_B = "./model/gen_b_new.pth"
CHECKPOINT_GEN_A = "./model/gen_a_new.pth"
CHECKPOINT_DISC_B = "./model/disc_b_new.pth"
CHECKPOINT_DISC_A = "./model/disc_a_new.pth"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
transforms_app = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)