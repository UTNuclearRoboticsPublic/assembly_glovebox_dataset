import glob2 as glob
from pathlib import Path
import PIL
import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as transforms


# class HADRDataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         # data_path,
#         batch_size=32,
#         num_workers=2,
#         pin_memory=True,
#     ):
#         super().__init__()
#         # self.data_path = data_path
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory

#     def setup(self, stage=None):
#         # Transformations applied to the dataset
#         trafo = transforms.Compose([transforms.Resize(size=(64, 64), interpolation=PIL.Image.NEAREST),
#                                 transforms.ToTensor()])

#         self.train_dataset = HADR(
#             transform=trafo,
#             # data_path=self.data_path,
#         )

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#         )


class HADR(torch.utils.data.Dataset):
    def __init__(self, transform, data_path="/home/slwanna/assembly_glovebox_dataset/Other_Datasets/HaDr/sim_train_320x256/"):
        # Initialization
        self.transform = transform
        # Image List
        self.img_root_path = data_path + "color"
        all_image_paths = sorted(glob.glob(self.img_root_path + "/*"))
        self.image_paths = all_image_paths

        self.label_root_path = data_path + "mask2"
        all_label_paths = sorted(glob.glob(self.label_root_path + "/*"))
        self.label_paths = all_label_paths


    def get_pathname(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        return image_path, label_path

    def __len__(self):
        # Returns the total number of samples in the DataSet
        return len(self.label_paths)

    def __getitem__(self, idx):

        # TODO: 
        # 1. Start by drawing a label file path
        label_path = self.label_paths[idx]

        # 2. Find the corresponding img... but how?
        # We know that the image filepath (excluding the *.png extension) will be a substring within the label
        # 2a. Drop the *.png
        image_substring = Path(label_path).stem
        if (image_substring[-3] == "_"):
            image_substring = image_substring[:-3]

        # print(f"label path: {label_path}")
        # print(f"image substring: {image_substring}")
        
        # 3. Find image substring in img_path list
        # image_path = self.image_paths[idx]
        image_path = [s for s in self.image_paths if image_substring in s][0]

        # 4. Do all this other stuff
        # print(f"img path: {image_path}")
        label_path = self.label_paths[idx]
        # print(f"label path: {label_path}")

        def path_to_image(img_path, label_path):
            img = Image.open(img_path)  # Read in with Pillow
            label = Image.open(label_path)  # Read in with Pillow
            return img, label

        # Prepares images
        def pipe(x,y):
            x, y = path_to_image(x, y)
            x = self.transform(x)
            y = self.transform(y)
            x = x.float()
            y = y.float()
            return x, y

        image, label = pipe(image_path, label_path)
        label = torch.squeeze(label)
        # label = torch.moveaxis(label, 0, -1)
        # scaled_labels = (label * 255).int()
        return image, [label.float(), label.float()]


if __name__ == "__main__":
    trafo = transforms.Compose([transforms.Resize(size=(64, 64), interpolation=PIL.Image.NEAREST),
                                transforms.ToTensor()])
    DATA_PATH = "/home/slwanna/assembly_glovebox_dataset/Other_Datasets/HaDr/sim_train_320x256/"
    dataset = HADR(trafo, data_path=DATA_PATH)

    # print(f"{len(dataset)} images found in {DATA_PATH} and shape is {dataset[0][0].shape}")
    # print(f"{len(dataset)} labels found in {DATA_PATH} and shape is {dataset[0][1].shape}")

    # I never ran this below. Assuming god is on my side for this one.
    #
    testing = torch.utils.data.DataLoader(
        dataset, batch_size=7, shuffle=False
    )
    
    for i, (images, masks) in enumerate(testing):
        for j in range(images.shape[0]):
            print(f"shape of image: {images[j].shape}")
            print(f"shape of labels: {masks[j].shape}")  
            print(f"label unique values: {torch.unique(masks[j])}")  
            break  
        break
    # for img, mask in testing:
    #     index = 3
    #     img = img[index,:3,:,:]
    #     print(f"img shape: {img.shape}")
    #     mask = mask[index,:,:,:]
    #     print(f"mask shape: {mask.shape}")
    #     print(torch.unique(mask))


    #     print(f"to scale {torch.unique(mask)*255}")

    #     viz_img = np.moveaxis(img.numpy(),0,2) #[:,:,:]
    #     print(f"viz_img shape: {viz_img.shape}")

    #     viz_label = np.moveaxis(mask.numpy(),0,2)
    #     print(f"viz_label shape: {viz_label.shape}")

    #     fig, axs = plt.subplots(1, 2)

    #     lil_size = 8
    #     axs[0].set_title("IMG", fontsize=lil_size)
    #     axs[0].imshow(viz_img) # top view
    #     axs[0].axis("off")
    #     axs[1].set_title("MASK", fontsize=lil_size)
    #     axs[1].imshow(viz_label) # side view
    #     axs[1].axis("off")
    #     # plt.savefig("test.png")
    #     break