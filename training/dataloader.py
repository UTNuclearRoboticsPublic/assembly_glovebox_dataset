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
        self.images = [file for file in os.listdir(path_to_images) if file.lower().endswith('.png')]
        self.masks = [file for file in os.listdir(path_to_labels) if file.lower().endswith('.png')]

    def __len__(self) -> int:
        # print(f"the imagees are {self.images}")
        return len(self.images)

    def __getitem__(self, index):
        img = f"{self.path_to_images}/{self.images[index]}"
        label = f"{self.path_to_labels}/{self.masks[index]}"
        
        image = Image.open(img)
        image = image.convert("RGB")

        mask = Image.open(label)
        print(np.array(mask).shape)

        img = self.transform(image)
        mask = self.transform(mask)
        
        # makes mask into size [height, width]
        mask = mask[0, :, :] + mask[1, :, :] + torch.mul(mask[2, :, :], 2)

        return img, mask
    


# test the loader with the below
# print(f"the imagees are {os.listdir('./masks')}")

# dataset = AssemblyDataset(path_to_labels='./data/Labels/Test_Subject_1/ood/J/Side_View', path_to_images='./data/images/Test_Subject_1/ood/J/Side_View')

# dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

# for i, (images, masks) in enumerate(dataloader):
#     masks = masks[0]
#     images = images[0]
#     plt.imshow(np.transpose(images, (1, 2, 0)))
#     # plt.imshow(np.reshape(masks, (masks.shape[2], masks.shape[1], masks.shape[0])), alpha=0.2)
#     plt.imshow(masks, alpha=0.5)
#     plt.show()