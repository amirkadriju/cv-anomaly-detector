import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class KolektorDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale mask

        resize_size = (1240, 500)  # (height, width)
        # Apply transforms
        transform_image = T.Compose([
            T.Resize(resize_size),
            T.ToTensor()
        ])

        transform_mask = T.Compose([
            T.Resize(resize_size, interpolation=Image.NEAREST),
            T.ToTensor()
        ])

        image = transform_image(image)
        mask = transform_mask(mask)

        # Binarize mask: 1 if any pixel is non-zero (anomaly), else 0
        label = 1 if torch.any(mask > 0) else 0

        return image, mask, label
