from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
import PIL
import os
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import cv2

class AssemblyDataset(Dataset):
    def __init__(self, path_to_images, path_to_1_labels, path_to_2_labels, img_size, gs_images=None, gs_labels=None):

        self.transform = self.get_transform(img_size)
        self.path_to_images = path_to_images

        # TODO: set two paths to make two self.masks so that there is one for each annotator

        self.path_to_1_labels = path_to_1_labels
        self.path_to_2_labels = path_to_2_labels

        self.images = [os.path.join(path, file) for path in path_to_images for file in os.listdir(path) if file.endswith('.png')]
        self.masks_1 = [os.path.join(path, file) for path in path_to_1_labels for file in os.listdir(path)]
        self.masks_2 = [os.path.join(path, file) for path in path_to_2_labels for file in os.listdir(path)]

        if gs_images and gs_labels:
            labels1, labels2 = gs_labels
            imgs = gs_images

            matching_masks_1 = [os.path.join(path, file) for path in labels1 for file in os.listdir(path) if 'NG_G' in file or 'GL_G' in file]
            matching_masks_2 = [os.path.join(path, file) for path in labels2 for file in os.listdir(path) if 'NG_G' in file or 'GL_G' in file]
            images = [os.path.join(path, file) for path in imgs for file in os.listdir(path) if file.endswith('.png')]

            self.masks_1.extend(matching_masks_1)
            self.masks_2.extend(matching_masks_2)
            self.images.extend(images)

        

    def __len__(self) -> int:
        # don't worry about this
        return len(self.images)

    def __getitem__(self, index):

        # TODO: change so mask is [y1, y2] where the first are the masks from the first annotator,
        # and the second are the masks from the second

        img = self.images[index]
        label_1 = self.masks_1[index]
        label_2 = self.masks_2[index]
        
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_1 = cv2.imread(label_1)
        mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_BGR2RGB)

        mask_2 = cv2.imread(label_2)
        mask_2 = cv2.cvtColor(mask_2, cv2.COLOR_BGR2RGB)


        # transform_input = {
        #     "image": image,
        #     "mask_1": mask_1,
        #     "mask_2": mask_2
        # }

        # output = self.transform(transform_input)

        # print(f"the output is {output}")

        assert img.split('/')[-1].split('.')[0] == label_1.split('/')[-1].split('.')[0] == label_2.split('/')[-1].split('.')[0], f"Label names and images do not match for index {img} and {label_1} and {label_2} and {self.images} and {self.masks_1}"

        
        
        transformed = self.transform(image=image, masks=[mask_1, mask_2])

        # assert that label names and images match

        img = transformed['image'] / 255
        mask_1, mask_2 = transformed['masks']

        mask_1 = mask_1.type(torch.float32)
        mask_2 = mask_2.type(torch.float32)
        img = img.type(torch.float32)

        # print(torch.unique(img))

        # print(f"the shape of mask_1 is {mask_1.shape} and {mask_2.shape} and {img.shape} and {mask_1.dtype}, {mask_2.dtype}, {img.dtype}, {torch.unique(mask_1)}")
        # img = self.transform(image)
        # mask_1 = self.transform(mask_1)
        # mask_2 = self.transform(mask_2)
        
        mask_1[mask_1 == 255] = 1
        mask_2[mask_2 == 255] = 1

        # print(f"the size of mask_1 is {mask_1.shape}")
        
        # makes mask into size [height, width] with respective class at each index
        def mask_to_2D(mask):
            mask = mask[0, :, :] + mask[1, :, :] + torch.mul(mask[2, :, :], 2)
            return mask
        
        
        mask_1 = mask_to_2D(mask_1)
        mask_2 = mask_to_2D(mask_2)

        # print(f"unique in mask_1 is {torch.unique(mask_1)}")

        return img, [mask_1, mask_2]
    
    def get_transform(self, img_size):
        transform = A.Compose ([
            A.Resize(height=img_size, width=img_size),
            A.ColorJitter(),
            A.AdvancedBlur(),
            A.GaussNoise(),
            A.RandomRotate90(p=0.5),
            ToTensorV2(transpose_mask=True)
        ])
        return transform


if __name__ == "__main__":
    # test the loader with the below

    dataset = AssemblyDataset(path_to_1_labels=['./Dataset/Annotations/Test_Subject_1/By_1/id/Jenga_task/Side_View'], 
                              path_to_2_labels=['./Dataset/Annotations/Test_Subject_1/By_2/id/Jenga_task/Side_View'],
                              path_to_images=['./Dataset/Sampled_Frames/Test_Subject_1/id/Jenga_task/Side_View'],
                              img_size=161)

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    for i, (images, masks) in enumerate(dataloader):
        masks = masks[0][0]
        images = images[0]
        plt.imshow(np.transpose(images, (1, 2, 0)))
        # plt.imshow(np.reshape(masks, (masks.shape[2], masks.shape[1], masks.shape[0])), alpha=0.2)
        plt.imshow(masks, alpha=0.5)
        plt.savefig("hi.png")