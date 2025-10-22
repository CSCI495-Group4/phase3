# train/test_forward.py
import torch
from models.efficientnet_baseline import EfficientNetBaseline
from models.multimodal_fer import MultiModalFER
from utils.config import NUM_CLASSES, DEVICE, MAX_TOKENS

def main():
    x_img = torch.randn(4, 3, 224, 224).to(DEVICE)
    # Baseline
    m1 = EfficientNetBaseline(num_classes=NUM_CLASSES).to(DEVICE)
    y1 = m1(x_img)
    print("Baseline logits:", y1.shape)  # (4, 7)

    # Multimodal
    m2 = MultiModalFER(num_classes=NUM_CLASSES).to(DEVICE)
    input_ids = torch.randint(1, 30000, (4, MAX_TOKENS)).to(DEVICE)
    attention_mask = torch.ones(4, MAX_TOKENS, dtype=torch.long).to(DEVICE)
    y2 = m2(x_img, input_ids, attention_mask)
    print("Multimodal logits:", y2.shape)  # (4, 7)

if __name__ == "__main__":
    main()
