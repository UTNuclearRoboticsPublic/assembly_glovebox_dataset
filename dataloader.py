from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import PIL
import os
import matplotlib.pyplot as plt

transform = transforms.Compose ([
    transforms.Resize(size=(161, 161), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor
])

class AssemblyDataset(Dataset):
    def __init__(self, path_to_images, path_to_labels, transform):
        self.transform = transform
        self.path_to_images = path_to_images
        self.path_to_labels = path_to_labels
        self.images = os.listdir(path_to_images).sort()
        self.masks = os.listdir(path_to_labels).sort()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index):
        img = os.path.join(self.path_to_images, self.images[index])
        label = os.path.join(self.path_to_labels, self.path_to_labels)

        image = Image.open(img)
        mask = Image.open(label)

        return self.transform(image), self.transform(mask)