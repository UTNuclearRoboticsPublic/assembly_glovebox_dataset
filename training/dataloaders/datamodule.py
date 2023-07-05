import lightning.pytorch as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
import os

from training.dataloaders.dataloader import AssemblyDataset


class AssemblyDataModule(pl.LightningDataModule):
    def __init__(self, fit_query, test_query):
        self.fit_query = fit_query
        self.test_query = test_query
        print(f"Initialized {fit_query} and {test_query}")
        super().__init__()

    def setup(self, stage: str):
        fit_files = []


        if stage=="test" or stage=="predict":
            path_to_imgs, path_to_labels = self._get_files(self.test_query)
            # self.test_set = AssemblyDataset(path_to_labels='./data/Labels/Test_Subject_1/id/J/Top_View', path_to_images='./data/images/Test_Subject_1/id/TB/Top_View')
            self.test_set = AssemblyDataset(path_to_labels=path_to_labels, path_to_images=path_to_imgs)

        if stage=="fit":
            path_to_imgs, path_to_labels = self._get_files(self.fit_query)
            train_set = AssemblyDataset(path_to_labels=path_to_labels, path_to_images=path_to_imgs)

            train_set_size = int(len(train_set)*0.8)
            valid_set_size = len(train_set) - train_set_size

            self.train_set, self.valid_set = random_split(train_set, [train_set_size, 
            valid_set_size])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=4, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=4, shuffle=False, num_workers=4)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=4, shuffle=False, num_workers=4)
    
    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=4, shuffle=False, num_workers=4)

    def _get_files(self, query):
        # appending all directory paths in the test_query
        label_dirs = []
        image_dirs = []
        for participant in query[0].split(", "):
            for dist in query[1].split(", "):
                for task in query[2].split(", "):
                    for view in query[3].split(", "):
                        label_path = os.path.join('.', 'data', 'Labels', participant, dist, task, view)
                        image_path = os.path.join('.', 'data', 'images', participant, dist, task, view)
                        label_dirs.append(label_path)
                        image_dirs.append(image_path)
        return image_dirs, label_dirs