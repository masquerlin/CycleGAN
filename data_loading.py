from torch.utils.data import Dataset
import os,config,torch
from PIL import Image
import numpy as np

class Dataset_loading(Dataset):
    def __init__(self, root_A, root_B, transform=None) -> None:
        super().__init__()
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform

        self.A_images = os.listdir(root_A)
        self.B_images = os.listdir(root_B)

        self.length_dataset = max(len(self.A_images), len(self.B_images))
        self.A_len = len(self.A_images)
        self.B_len = len(self.B_images)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        A_img = self.A_images[index % self.A_len]
        B_img = self.B_images[index % self.B_len]

        A_path = os.path.join(self.root_A, A_img)
        B_path = os.path.join(self.root_B, B_img)

        A_img = np.array(Image.open(A_path).convert("RGB"))
        B_img = np.array(Image.open(B_path).convert("RGB"))
        # print(type(A_img))
        # print(A_img.shape)

        if self.transform:
            augmentations = self.transform(image=A_img, image0=B_img)
            A_img = augmentations['image']
            B_img = augmentations['image0']
        # print(type(A_img))
        # print(A_img.shape)
        return A_img, B_img
# dataset = Dataset_loading(config.TRAIN_DIR_A, config.TRAIN_DIR_B, transform=config.transforms)
# for x,y in dataset:
#     print(x.shape)
#     break