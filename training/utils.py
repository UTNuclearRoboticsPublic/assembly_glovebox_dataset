import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataloader import AssemblyDataset
from torch.utils.data import DataLoader

def get_loaders(batch_size):
    
    train_ds = AssemblyDataset(path_to_labels='./data/Trash/masks', path_to_images='./data/Trash/images')
    train_loader = DataLoader(dataset=train_ds, batch_size = batch_size, num_workers=4, shuffle = True)

    val_ds = AssemblyDataset(path_to_labels='./data/Trash/masks', path_to_images='./data/Trash/images')
    val_loader = DataLoader(dataset=val_ds, batch_size = batch_size, num_workers=4, shuffle = False)

    return train_loader, val_loader


def save_predictions_as_imgs(train_set, loader, model, raw_images):
    model.eval()
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        outputs = model(data)

        with torch.no_grad():
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1).detach().cpu()
    
    # now save predictions and allow ability to go back and find the raw images
    # and compare the two
        
        