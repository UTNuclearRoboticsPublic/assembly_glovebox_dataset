import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
import os
import PIL
import torchvision.transforms as transforms
from transfer_learning.tl_dataloaders.HaDR_dataloader import HADR
from transfer_learning.tl_dataloaders.bin_dataloader import BinAssemblyDataset
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

class BinAssemblyHaDRDataModule(pl.LightningDataModule):
    def __init__(self, fit_query, test_query, batch_size, img_size):
        self.fit_query = fit_query
        self.test_query = test_query
        self.batch_size = batch_size
        self.img_size = img_size
        super().__init__()

    def setup(self, stage: str):
        fit_files = []

        if stage=="test" or stage=="predict":
            path_to_imgs, path_to_1_labels, path_to_2_labels, gs_images, (gs_labels_1, gs_labels_2) = self._get_files(self.test_query)
            # print(f"label 1 paths: {path_to_1_labels}")
            # print(f"label 2 paths: {path_to_2_labels}")
            # print(f"img paths: {path_to_imgs}")

            self.test_set = BinAssemblyDataset(path_to_1_labels=path_to_1_labels, path_to_2_labels = path_to_2_labels, path_to_images=path_to_imgs, img_size=self.img_size, gs_images=gs_images, gs_labels=(gs_labels_1, gs_labels_2))

        if stage=="fit":
            trafo = transforms.Compose([transforms.Resize(size=(256, 256), interpolation=PIL.Image.NEAREST),
                                transforms.ToTensor()])

            train_set = HADR(
                transform=trafo,
                # data_path=self.data_path,
            )
            # path_to_imgs, path_to_1_labels, path_to_2_labels, _, _ = self._get_files(self.fit_query)
            # # print(f"label 1 paths: {path_to_1_labels}")
            # # print(f"label 2 paths: {path_to_2_labels}")
            # # print(f"img paths: {path_to_imgs}")
            # train_set = BinAssemblyDataset(path_to_1_labels=path_to_1_labels, path_to_2_labels=path_to_2_labels, path_to_images=path_to_imgs, img_size=self.img_size)

            train_set_size = int(len(train_set)*0.89) # left with ~ 80/10/10 split (left one participant for the test set ID)
            valid_set_size = len(train_set) - train_set_size

            self.train_set, self.valid_set = random_split(train_set, [train_set_size, valid_set_size])

            # # changing val set to the test query
            
            # self.train_set = train_set

            # # val_path_to_imgs, val_path_to_1_labels, val_path_to_2_labels = self._get_files(self.test_query)
            # self.valid_set = AssemblyDataset(path_to_1_labels=val_path_to_1_labels, path_to_2_labels=val_path_to_2_labels, path_to_images=val_path_to_imgs, img_size=self.img_size)


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=128, pin_memory=True, worker_init_fn=set_worker_sharing_strategy)
    
    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, num_workers=128, pin_memory=True, worker_init_fn=set_worker_sharing_strategy)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=128, pin_memory=True, worker_init_fn=set_worker_sharing_strategy)
    
    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=128, pin_memory=True, worker_init_fn=set_worker_sharing_strategy)

    def _get_files(self, query):
        # appending all directory paths in the test_query

        # set the split queries here
        # if it is All, then set it to all possible queries for that particular solution

        label_1_dirs = []
        label_2_dirs = []
        image_dirs = []
        gs_labels_1 = []
        gs_labels_2 = []
        gs_images = []

        # print(f"the query is {query}")

        SELMA_IMG_PREFIX_PATH = "/home/slwanna/FIRETEAM/Data/HAGS/FINAL_SAMPLED_FRAMES"
        SELMA_LABEL_PREFIX_PATH = "/home/slwanna/FIRETEAM/Data/HAGS/FINAL_ANNOTATIONS"

        try: 
            x = query.get("participants")
        except:
            print(f"the query is {query}")
            query = dict(query)
            # query["participants"] = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
        for participant in query.get("participants"):
            for dist in query.get("distribution"):
                for task in query.get("task"):
                    for view in query.get("view"):
                        # TODO: create two paths -> one for first annotator, one for the second
                        # I added a part in the parser so that the annotator number will only be either 1 or 2
                        image_path = os.path.join(SELMA_IMG_PREFIX_PATH, participant, dist, task, view)

                        if dist=="replaced_green_screen":
                            label_1_path = os.path.join(SELMA_LABEL_PREFIX_PATH, participant, 'By_1', "ood", task, view)
                            label_2_path = os.path.join(SELMA_LABEL_PREFIX_PATH, participant, 'By_2', "ood", task, view)
                            gs_labels_1.append(label_1_path)
                            gs_labels_2.append(label_2_path)
                            gs_images.append(image_path)

                        else:
                            dist2 = "ood" if dist=="replaced_green_screen" else dist
                            label_1_path = os.path.join(SELMA_LABEL_PREFIX_PATH, participant, 'By_1', dist2, task, view)
                            label_2_path = os.path.join(SELMA_LABEL_PREFIX_PATH, participant, 'By_2', dist2, task, view)

                            label_1_dirs.append(label_1_path)
                            label_2_dirs.append(label_2_path)
                        
                            image_dirs.append(image_path)
        # if gs_labels is empty, set to none
        if len(gs_labels_1)==0:
            gs_labels_1, gs_labels_2, gs_images = None, None, None
        return image_dirs, label_1_dirs, label_2_dirs, gs_images, (gs_labels_1, gs_labels_2)