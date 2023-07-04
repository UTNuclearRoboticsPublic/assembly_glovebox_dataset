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

        self.images = [os.path.join(path, file) for path in path_to_images for file in os.listdir(path) if file.endswith('.png')]
        self.masks = [os.path.join(path, file) for path in path_to_labels for file in os.listdir(path)]



    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index):

        img = self.images[index]
        label = self.masks[index]
        
        image = Image.open(img)
        image = image.convert("RGB")

        mask = Image.open(label)

        img = self.transform(image)
        mask = self.transform(mask)
        
        # makes mask into size [height, width] with respective class at each index
        mask = mask[0, :, :] + mask[1, :, :] + torch.mul(mask[2, :, :], 2)

        return img, mask
    

if __name__ == "__main__":
    # test the loader with the below

    dataset = AssemblyDataset(path_to_labels=['.\\data\\Labels\\Test_Subject_1\\id\\J\\Side_View', '.\\data\\Labels\\Test_Subject_1\\id\\J\\Top_View'], 
                              path_to_images=['.\\data\\Labels\\Test_Subject_1\\id\\J\\Side_View', '.\\data\\Labels\\Test_Subject_1\\id\\J\\Top_View'])

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    for i, (images, masks) in enumerate(dataloader):
        masks = masks[0]
        images = images[0]
        plt.imshow(np.transpose(images, (1, 2, 0)))
        # plt.imshow(np.reshape(masks, (masks.shape[2], masks.shape[1], masks.shape[0])), alpha=0.2)
        plt.imshow(masks, alpha=0.5)
        plt.show()