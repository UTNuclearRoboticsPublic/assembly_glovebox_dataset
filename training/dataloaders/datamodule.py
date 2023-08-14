import lightning.pytorch as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
import os

from training.dataloaders.dataloader import AssemblyDataset


class AssemblyDataModule(pl.LightningDataModule):
    def __init__(self, fit_query, test_query, batch_size, img_size):
        self.fit_query = fit_query
        self.test_query = test_query
        self.batch_size = batch_size
        self.img_size = img_size
        print(f"Initialized {fit_query} and {test_query}")
        super().__init__()

    def setup(self, stage: str):
        fit_files = []

        if stage=="test" or stage=="predict":
            path_to_imgs, path_to_1_labels, path_to_2_labels = self._get_files(self.test_query)
            self.test_set = AssemblyDataset(path_to_1_labels=path_to_1_labels, path_to_2_labels = path_to_2_labels, path_to_images=path_to_imgs, img_size=self.img_size)

        if stage=="fit":
            path_to_imgs, path_to_1_labels, path_to_2_labels = self._get_files(self.fit_query)
            train_set = AssemblyDataset(path_to_1_labels=path_to_1_labels, path_to_2_labels=path_to_2_labels, path_to_images=path_to_imgs, img_size=self.img_size)

            train_set_size = int(len(train_set)*0.89) # left with ~ 80/10/10 split (left one participant for the test set ID)
            valid_set_size = len(train_set) - train_set_size

            self.train_set, self.valid_set = random_split(train_set, [train_set_size, 
            valid_set_size])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def _get_files(self, query):
        # appending all directory paths in the test_query

        # set the split queries here
        # if it is All, then set it to all possible queries for that particular solution

        label_1_dirs = []
        label_2_dirs = []
        image_dirs = []
        for participant in query.get("participants"):
            for dist in query.get("distribution"):
                for task in query.get("task"):
                    for view in query.get("view"):
                        # TODO: create two paths -> one for first annotator, one for the second
                        # I added a part in the parser so that the annotator number will only be either 1 or 2
                        label_1_path = os.path.join('.', 'data', 'Labels', participant, 'By_1', dist, task, view)
                        label_2_path = os.path.join('.', 'data', 'Labels', participant, 'By_2', dist, task, view)
                        image_path = os.path.join('.', 'data', 'images', participant, dist, task, view)
                        label_1_dirs.append(label_1_path)
                        label_2_dirs.append(label_2_path)
                        image_dirs.append(image_path)
        return image_dirs, label_1_dirs, label_2_dirs