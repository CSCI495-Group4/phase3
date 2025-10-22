# models/efficientnet_baseline.py
import torch.nn as nn
import timm

class EfficientNetBaseline(nn.Module):
    def __init__(self, num_classes=7, backbone='efficientnet_b0'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=num_classes, in_chans=3)
        # timm will insert the classifier head; no need to manually slice
    def forward(self, x):
        return self.backbone(x)
