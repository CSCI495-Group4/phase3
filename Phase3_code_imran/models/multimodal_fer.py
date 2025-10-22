# models/multimodal_fer.py
import torch
import torch.nn as nn
from torchvision import models
from utils.config import VOCAB_SIZE, MAX_TOKENS

class TinyTextEncoder(nn.Module):
    """
    Lightweight text encoder to avoid heavy deps until real text is ready.
    Token Embedding -> 1D Conv -> Global Max Pool -> Linear
    """
    def __init__(self, vocab_size=VOCAB_SIZE, emb_dim=128, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = nn.Conv1d(emb_dim, hidden, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc   = nn.Linear(hidden, 256)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (B, L)
        x = self.emb(input_ids).transpose(1, 2)  # (B, E, L)
        x = torch.relu(self.conv(x))             # (B, H, L)
        x = self.pool(x).squeeze(-1)             # (B, H)
        x = self.fc(x)                            # (B, 256)
        return x

class MultiModalFER(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # Image encoder: ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # (B,512,1,1)
        self.cnn_fc = nn.Linear(512, 256)

        # Text encoder: tiny conv text net (swap to BERT later if needed)
        self.text_encoder = TinyTextEncoder()

        # Fusion + classifier
        self.fuse = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.cls = nn.Linear(128, num_classes)

    def forward(self, img, input_ids, attention_mask=None):
        # Image path
        f_img = self.cnn(img).flatten(1)   # (B,512)
        f_img = self.cnn_fc(f_img)         # (B,256)

        # Text path
        f_txt = self.text_encoder(input_ids, attention_mask)  # (B,256)

        # Fusion
        z = torch.cat([f_img, f_txt], dim=1)  # (B,512)
        z = self.fuse(z)                      # (B,128)
        logits = self.cls(z)                  # (B,7)
        return logits
