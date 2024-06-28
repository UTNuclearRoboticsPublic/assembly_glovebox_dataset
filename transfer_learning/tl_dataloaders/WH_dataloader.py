import glob2 as glob
import PIL
import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# class WHDataModule(pl.LightningDataModule):
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
#         self.train_dataset = WH(
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


class WH(torch.utils.data.Dataset):
    def __init__(self, transform, data_path="Other_Datasets/WorkingHands/",):
        # Initialization
        self.transform = transform
        
        # Real Data
        self.real_img_root_path = data_path + "real_data/real_data"
        all_real_image_paths = sorted(glob.glob(self.real_img_root_path + "/*/*/*/*.png"))
        self.real_image_paths = all_real_image_paths
        self.real_image_paths = [i for i in self.real_image_paths if "RGBImages/" in i]
        # print(len(self.real_image_paths))

        self.real_label_root_path = self.real_img_root_path
        all_real_label_paths = sorted(glob.glob(self.real_label_root_path + "/*/*/*/*.jpg"))
        self.real_label_paths = all_real_label_paths
        self.real_label_paths = [i for i in self.real_label_paths if "SegmentationClass/" in i]
        # print(len(self.real_label_paths))

        # Synth Data synth
        self.synth_img_root_path = data_path + "syn_data"
        all_synth_image_paths = sorted(glob.glob(self.synth_img_root_path + "/*/*/*/*.png"))
        self.synth_image_paths = all_synth_image_paths
        self.synth_image_paths = [i for i in self.synth_image_paths if "RGBImages/" in i]
        # print(len(self.synth_image_paths))

        self.synth_label_root_path = self.synth_img_root_path
        all_synth_label_paths = sorted(glob.glob(self.synth_label_root_path + "/*/*/*/*.png"))
        self.synth_label_paths = all_synth_label_paths
        self.synth_label_paths = [i for i in self.synth_label_paths if "SegmentationClass/" in i]
        # print(len(self.synth_label_paths))

        # 3695*0.8 = 2956 
        # 4210*0.8 = 3368
        self.real_img_train = self.real_image_paths[:2956]
        self.real_label_train = self.real_label_paths[:2956]
        self.synth_img_train = self.synth_image_paths[:3368]
        self.synth_label_train = self.synth_label_paths[:3368]

        self.image_paths = self.real_img_train + self.synth_img_train
        self.label_paths = self.real_label_train + self.synth_label_train

    def get_pathname(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        return image_path, label_path

    def __len__(self):
        # Returns the total number of samples in the DataSet
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        # print(label_path)

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
            y = torch.round(y.float()*255)
            y[y!=1]=0
            return x, y

        image, label = pipe(image_path, label_path)
        label = torch.squeeze(label)
        # print(f'Label shape: {label.shape}')
        fixed_img = image[:3,:,:]
        # scaled_label = label * 255 
        # scaled_label = torch.moveaxis(scaled_label, 0, -1).int()
        return fixed_img, [label.float(), label.float()]


if __name__ == "__main__":
    # From paper: 
    # The real part of the dataset has 3695 labeled images, while in the synthetic part has 4170 images.

    trafo = transforms.Compose([transforms.Resize(size=(64, 64), interpolation=PIL.Image.NEAREST),
                                transforms.ToTensor()])
    DATA_PATH = "Other_Datasets/WorkingHands/"
    dataset = WH(trafo, data_path=DATA_PATH)

    # print(f"{len(dataset)} images found in {DATA_PATH} and shape is {dataset[0][0].shape}")
    # print(f"{len(dataset)} labels found in {DATA_PATH} and shape is {dataset[0][1].shape}")

    # I never ran this below. Assuming god is on my side for this one.
    #
    testing = torch.utils.data.DataLoader(
        dataset, batch_size=20, shuffle=False
    )
    for i, (images, masks) in enumerate(testing):
        for j in range(images.shape[0]):
            print(f"shape of image: {images.shape}")
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


    #     # print(f"to scale {torch.unique(mask)*255}")
    #     print(f"to scale {torch.unique(mask)}")

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

