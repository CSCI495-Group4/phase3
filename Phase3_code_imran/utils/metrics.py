# utils/metrics.py
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

@torch.no_grad()
def evaluate_logits(logits, labels, average="macro"):
    preds = logits.argmax(dim=1).cpu().numpy()
    y = labels.cpu().numpy()
    acc = accuracy_score(y, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average=average, zero_division=0)
    cm = confusion_matrix(y, preds)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm}
