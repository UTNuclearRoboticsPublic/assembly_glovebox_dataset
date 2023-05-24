import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
from dataloader import AssemblyDataset

class AssemblyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
    def setup(self, stage: str):

        if stage=="test" or stage=="predict":
            self.test_set = AssemblyDataset(path_to_labels='./data/Labels/Test_Subject_1/ood/J/Side_View', path_to_images='./data/images/Test_Subject_1/ood/J/Side_View')
        
        if stage=="fit":
            train_set = AssemblyDataset(path_to_labels='./data/Labels/Test_Subject_1/ood/J/Side_View', path_to_images='./data/images/Test_Subject_1/ood/J/Side_View')

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

        