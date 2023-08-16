from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms.v2 as transforms
import numpy as np
from PIL import Image
import PIL
import os
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AssemblyDataset(Dataset):
    def __init__(self, path_to_images, path_to_1_labels, path_to_2_labels, img_size):

        self.transform = self.get_transform(img_size)
        self.path_to_images = path_to_images

        # TODO: set two paths to make two self.masks so that there is one for each annotator

        self.path_to_1_labels = path_to_1_labels
        self.path_to_2_labels = path_to_2_labels

        self.images = [os.path.join(path, file) for path in path_to_images for file in os.listdir(path) if file.endswith('.png')]
        self.masks_1 = [os.path.join(path, file) for path in path_to_1_labels for file in os.listdir(path)]
        self.masks_2 = [os.path.join(path, file) for path in path_to_2_labels for file in os.listdir(path)]

    def __len__(self) -> int:
        # don't worry about this
        return len(self.images)

    def __getitem__(self, index):

        # TODO: change so mask is [y1, y2] where the first are the masks from the first annotator,
        # and the second are the masks from the second

        img = self.images[index]
        label_1 = self.masks_1[index]
        label_2 = self.masks_2[index]
        
        image = Image.open(img)
        image = image.convert("RGB")

        mask_1 = Image.open(label_1)
        mask_2 = Image.open(label_2)

        # transform_input = {
        #     "image": image,
        #     "mask_1": mask_1,
        #     "mask_2": mask_2
        # }

        # output = self.transform(transform_input)

        # print(f"the output is {output}")

        jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        image = jitter(image)

        img, mask_1, mask_2 = self.transform(image, mask_1, mask_2)

        # img = self.transform(image)
        # mask_1 = self.transform(mask_1)
        # mask_2 = self.transform(mask_2)
        
        # makes mask into size [height, width] with respective class at each index
        def mask_to_2D(mask):
            mask = mask[0, :, :] + mask[1, :, :] + torch.mul(mask[2, :, :], 2)
            return mask
        
        mask_1 = mask_to_2D(mask_1)
        mask_2 = mask_to_2D(mask_2)

        return img, [mask_1, mask_2]
    
    def get_transform(self, img_size):
        transform = A.Compose ([
            A.Resize(height=img_size, width=img_size, interpolation=PIL.Image.NEAREST),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])
        return transform


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