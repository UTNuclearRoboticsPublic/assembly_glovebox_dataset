import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import cv2
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# cd_str = "/Users/slwanna/vscode/datasets/HRC_DS/"
# new_H =180
# new_W = 320
# data_type = "agent"

# MFW_data=np.load(cd_str+'ZL_dataset/'+str(new_H)+'x'+str(new_W)+'_'+data_type+'_MFW_data.npz')

# pixel_unique=MFW_data['pixel_unique']
# pixel_count=MFW_data['pixel_count']
# class_weights_freq=MFW_data['class_weights_freq']

# print('pixel_unique:')
# print(pixel_unique)

# print('pixel_count:')
# pixel_count[1]=sum(pixel_count[1:])
# pixel_count=pixel_count[0:5]
# print(pixel_count)
# pixel_med=np.median(pixel_count)
# class_weights_freq=pixel_med/pixel_count

# class_weights_freq=class_weights_freq*0+1
# print('class_weights_freq:')
# print(class_weights_freq)

# pkl_file = open('/home/slwanna/FIRETEAM/Data/HRC_DS/ZL_dataset/Big_data_dictionary'+'.pkl', 'rb')
# data = pickle.load(pkl_file)

# data.keys()

class ZL(torch.utils.data.Dataset):
# class ZL_data_split:
    
    def __init__(self, split_id, data_path='./Other_Datasets/HRC_DS/ZL_dataset/Big_data_dictionary.pkl',):
                #  data_split_dict,split_type,split_id,
                #  new_H=new_H,new_W=new_W):
        # self.transform = transform
        # self.data_split_dict=data_split_dict
        self.data_path = data_path
        self.index_split = 1 # change this
        self.split_id=split_id      # 'train', 'val', 'test'
        self.split_types=  ['tool', 'agent', 'setup', 'random']
        # self.split_key = self.split_types[0]
        self.new_H = 180
        self.new_W = 320
        pkl_file = open(self.data_path, 'rb')
        self.data = pickle.load(pkl_file)
        split_key=self.split_id+'_indx_bin'
        self.all_data = self.data["tool"][split_key] + self.data["agent"][split_key] + self.data["setup"][split_key] + self.data["random"][split_key]
    
    def __len__(self):
        # Returns the total number of samples in the DataSet
        split_key=self.split_id+'_indx_bin'

        return len(self.all_data)

    def __getitem__(self, idx, resize=True,verbose=0,
                  norm_RGB=255):
        
        # print(self.all_data)
        # vid_frm_id_bin=self.data[self.split_types[self.index_split]][split_key][idx]
        vid_frm_id_bin=self.all_data[idx]

        data_tmp=np.load(os.path.join('Other_Datasets/HRC_DS/ZL_dataset/proc_'+str(self.new_H)+'x'+str(self.new_W)\
                                                    +'/'+vid_frm_id_bin+'.npz'))
        img_i=data_tmp['X']/norm_RGB
        # print(f"STARTINGSTARTINGSTARTINGSTARTING HERE")
        # print(f"img shape: {img_i.shape}")

        mask_i=data_tmp['Y']
        # print(f"mask type: {type(mask_i)}")

        # print(f"mask shape: {mask_i.shape}")
        # print(f"ENDINGENDINGENGINER HERE")

        mask = cv2.resize(np.array(mask_i), (256,256), interpolation =cv2.INTER_NEAREST)
        img = cv2.resize(np.array(img_i), (256,256), interpolation =cv2.INTER_NEAREST)

        mask =torch.from_numpy(mask).float()
        img =torch.moveaxis(torch.from_numpy(img), -1, 0).float()

        # print(f"img shape: {img.shape}")
        # print(f"img values: {img.unique()}")

        # print(f"mask type: {type(mask)}")
        # print(f"mask shape: {mask.shape}")
        # print(f"mask values: {mask.unique()}")

        if verbose==1:
            print('img_i.shape:  '+str(img_i.shape))
            print('mask_i.shape: '+str(mask_i.shape))
            
        return img , [mask, mask]  


if __name__ == "__main__":
    # From paper: 
    # The real part of the dataset has 3695 labeled images, while in the synthetic part has 4170 images.

    # trafo = transforms.Compose([transforms.Resize(size=(256, 256), interpolation=PIL.Image.NEAREST),
    #                             transforms.ToTensor()])
    # DATA_PATH = "/home/slwanna/FIRETEAM/Data/WorkingHands/"
    dataset = ZL("train")
    print(len(dataset))

    # print(f"{len(dataset)} images found in {DATA_PATH} and shape is {dataset[0][0].shape}")
    # print(f"{len(dataset)} labels found in {DATA_PATH} and shape is {dataset[0][1].shape}")

    # I never ran this below. Assuming god is on my side for this one.
    #
    testing = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False
    )
    for i, (images, masks) in enumerate(testing):
        for j in range(images.shape[0]):
            print(f"shape of image: {images.shape}")
            print(f"shape of labels: {masks[j].shape}")  
            # print(f"label unique values: {torch.unique(masks[j])}")  
            break
        break     

# training_set=ZL_data_split(data,data_type,'train',new_W=new_W,new_H=new_H)
# val_set=ZL_data_split(data,data_type,'val',new_W=new_W,new_H=new_H)
# test_set=ZL_data_split(data,data_type,'test',new_W=new_W,new_H=new_H)