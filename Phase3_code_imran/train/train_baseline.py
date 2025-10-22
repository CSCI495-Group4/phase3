# train/train_baseline.py
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
from data.dataloader_image import get_image_loaders
from models.efficientnet_baseline import EfficientNetBaseline
from utils.config import DEVICE, EPOCHS, LR, NUM_CLASSES
from utils.metrics import evaluate_logits

def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    running_loss, correct, n = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
        running_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += imgs.size(0)
    return running_loss/n, correct/n

@torch.no_grad()
def validate(model, loader, loss_fn):
    model.eval()
    running_loss, correct, n = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Valid"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += imgs.size(0)
    return running_loss/n, correct/n

def main():
    train_loader, val_loader, _ = get_image_loaders()
    model = EfficientNetBaseline(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn)
        va_loss, va_acc = validate(model, val_loader, loss_fn)
        print(f"Epoch {epoch}/{EPOCHS} | Train: loss {tr_loss:.4f}, acc {tr_acc:.4f} | Val: loss {va_loss:.4f}, acc {va_acc:.4f}")

if __name__ == "__main__":
    main()
