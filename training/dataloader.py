from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import PIL
import os
import matplotlib.pyplot as plt

transform = transforms.Compose ([
    transforms.Resize(size=(161, 161), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor()
])

class AssemblyDataset(Dataset):
    def __init__(self, path_to_images, path_to_labels):
        self.transform = transform
        self.path_to_images = path_to_images
        self.path_to_labels = path_to_labels
        self.images = os.listdir(path_to_images)
        self.masks = os.listdir(path_to_labels)

    def __len__(self) -> int:
        # print(f"the imagees are {self.images}")
        return len(self.images)

    def __getitem__(self, index):
        img = f"{self.path_to_images}/{self.images[index]}"
        label = f"{self.path_to_labels}/{self.masks[index]}"

        print(img)

        image = Image.open(img)
        image = image.convert("RGB")

        mask = Image.open(label)
        print(np.array(mask).shape)

        img = self.transform(image)
        mask = self.transform(mask)
        
        # makes mask into size [height, width]
        mask = mask[0, :, :] + mask[1, :, :] + torch.mul(mask[2, :, :], 2)

        return img, mask
    
# print(f"the imagees are {os.listdir('./masks')}")

dataset = AssemblyDataset(path_to_labels='./data/Trash/masks', path_to_images='./data/Trash/images')

dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

for i, (images, masks) in enumerate(dataloader):
    plt.imshow(np.transpose(images[0]))
    plt.imshow(np.reshape(masks, (masks.shape[2], masks.shape[1], masks.shape[0])))
    plt.show()