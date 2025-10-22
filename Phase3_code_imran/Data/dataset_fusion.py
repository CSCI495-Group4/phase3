# data/dataset_fusion.py
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.transforms import train_tfms, valid_tfms
from utils.config import DATA_ROOT, BATCH_SIZE, NUM_WORKERS
from data.dataloorer_text import dummy_tokenize
import torch

class ImageOnlyToMultimodal:
    """
    Wraps an ImageFolder dataset and adds dummy text inputs.
    """
    def __init__(self, base_ds, is_train=False):
        self.ds = base_ds
        self.is_train = is_train

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        # We'll add text later; for now, make dummy tokens per sample
        text_inputs = dummy_tokenize(batch_size=1)
        # squeeze batch dim → (L,)
        text_inputs = {k: v.squeeze(0) for k,v in text_inputs.items()}
        return img, text_inputs, label

def get_multimodal_loaders():
    train_base = datasets.ImageFolder(f"{DATA_ROOT}/train", transform=train_tfms)
    val_base   = datasets.ImageFolder(f"{DATA_ROOT}/val",   transform=valid_tfms)

    train_mm = ImageOnlyToMultimodal(train_base, is_train=True)
    val_mm   = ImageOnlyToMultimodal(val_base,   is_train=False)

    def collate_mm(batch):
        # batch: list of (img, text_inputs (dict), label)
        imgs  = torch.stack([b[0] for b in batch], dim=0)
        lbls  = torch.tensor([b[2] for b in batch], dtype=torch.long)
        # stack dict items
        input_ids     = torch.stack([b[1]["input_ids"]     for b in batch], dim=0)
        attention_mask= torch.stack([b[1]["attention_mask"]for b in batch], dim=0)
        text_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        return imgs, text_inputs, lbls

    train_loader = DataLoader(train_mm, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, collate_fn=collate_mm)
    val_loader   = DataLoader(val_mm,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_mm)
    return train_loader, val_loader
