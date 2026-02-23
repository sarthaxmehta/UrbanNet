import torch
from torch.utils.data import Dataset
import numpy as np


class BuildingDataset(Dataset):
    """
    PyTorch Dataset for building segmentation.
    Expects:
        images: numpy array (N, H, W, C)
        masks:  numpy array (N, H, W, 1)
    """

    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Convert to tensor and change to (C, H, W)
        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).permute(2, 0, 1).float()

        return image, mask
