# data/dataloader_image.py
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.transforms import train_tfms, valid_tfms
from utils.config import DATA_ROOT, BATCH_SIZE, NUM_WORKERS

def get_image_loaders():
    train_ds = datasets.ImageFolder(f"{DATA_ROOT}/train", transform=train_tfms)
    val_ds   = datasets.ImageFolder(f"{DATA_ROOT}/val",   transform=valid_tfms)
    test_ds  = datasets.ImageFolder(f"{DATA_ROOT}/test",  transform=valid_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader
