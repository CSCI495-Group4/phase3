# Multimodal Emotion Recognition (Image + Text Fusion Model)


## Imports 
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import timm
from datasets import load_dataset
from tqdm.notebook import tqdm
```
## Architecture Specifications
## --------------------------------------------------------------
### Image Backbone:  ResNet-18 (pretrained) → 512-D image feature
### Text Encoder:    GRU (RNN) → 512-D text embedding
### Fusion:          Concatenate [512 + 512] → 1024-D
### Dropout:         p = 0.5
### Head:            Linear(1024 → 7) + Softmax
### Loss:            CrossEntropy

## Text Dataset (Hugging Face Emotion) 

```python
emotion_dataset = load_dataset("emotion")
df_emotion = emotion_dataset['train'].to_pandas()
print(f"✓ Loaded text emotion dataset: {len(df_emotion)} samples")
```

## Image Transform (100×100 grayscale → 224×224 RGB) 
```python
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
```
## Image Encoder
```python
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = timm.create_model('resnet18', pretrained=True)
        self.feature = nn.Sequential(*list(base.children())[:-1])
        self.out_dim = 512
    def forward(self, x):
        return self.feature(x).flatten(1)
```
## Text Encoder 
```python
class TextEncoder(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.out_dim = hidden_dim
    def forward(self, x):
        _, h = self.gru(self.embedding(x))
        return h[-1]
```
## Fusion Model
```python
class MultiModalFER(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.txt_enc = TextEncoder()
        self.dropout = nn.Dropout(0.5)
        self.fuse = nn.Linear(self.img_enc.out_dim + self.txt_enc.out_dim, 1024)
        self.cls = nn.Linear(1024, num_classes)
    def forward(self, img, txt):
        f_img = self.img_enc(img)
        f_txt = self.txt_enc(txt)
        z = torch.cat([f_img, f_txt], dim=1)          # 🔥 Fusion happens here
        z = self.dropout(torch.relu(self.fuse(z)))
        return self.cls(z)
```
## Model Creation
```python
model = MultiModalFER(num_classes=7)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model ready on device:", device)
```
## Dummy Test Run 
```python
img = torch.randn(8, 3, 224, 224).to(device)
txt = torch.randint(0, 5000, (8, 20)).to(device)
out = model(img, txt)
print("Output shape:", out.shape)   # Expected: [8, 7]
```