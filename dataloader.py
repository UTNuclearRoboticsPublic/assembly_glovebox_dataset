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

transform2 = transforms.Compose ([
    # transforms.Resize(size=(64, 64), interpolation=PIL.Image.NEAREST),
    transforms.Resize(size=(161, 161), interpolation=PIL.Image.NEAREST),
])

class AssemblyDataset(Dataset):
    def __init__(self, path_to_images, path_to_labels, transform):
        self.transform = transform
        self.transform2 = transform2
        self.path_to_images = path_to_images
        self.path_to_labels = path_to_labels
        self.images = os.listdir(path_to_images)
        self.masks = os.listdir(path_to_labels)

    def __len__(self) -> int:
        # print(f"the imagees are {self.images}")
        return len(self.images)

    def __getitem__(self, index):
        img = f"{self.path_to_images}/{self.images[0]}"
        label = f"{self.path_to_labels}/{self.masks[0]}"

        print(img)

        image = Image.open(img)
        mask = Image.open(label)

        img = self.transform(image)
        mask = self.transform(mask)


        mask[np.where(mask<=0.5)] = 0
        mask[np.where(mask>0.5)] = 1

        print(f"unique values are {torch.unique(mask)}")


        return img, mask
    
# print(f"the imagees are {os.listdir('./masks')}")

dataset = AssemblyDataset('./images', './masks', transform)

dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

for i, (images, masks) in enumerate(dataloader):
    plt.imshow(np.transpose(images[0]))
    plt.imshow(np.transpose(masks[0]))
    plt.show()