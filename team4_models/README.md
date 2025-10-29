# Team 4 Phase 3 Models

This directory contains two different neural network architectures for emotion recognition from images and text metadata.

## Models Overview

### Architecture A (`team4phase3modela.ipynb`)
- **Image Input**: 100×100 grayscale → 224×224 RGB (ResNet-18 compatible)
- **Image Backbone**: ResNet-18 (pretrained) → 512-D image feature
- **Text Input**: Short text metadata (tokenized)
- **Text Encoder**: GRU (RNN) → 512-D text embedding
- **Fusion**: Concatenate [512-D image, 512-D text] → 1024-D
- **Dropout**: p=0.5 (randomly drops 50% of fused features during training)
- **Head**: Linear (1024 → 7), Softmax for probabilities
- **Loss**: Cross-Entropy

### Architecture B (`team4phase3modelb.ipynb`)
- **Image Input**: 100×100 grayscale → 224×224 RGB (ResNet-18 compatible)
- **Image Backbone**: ResNet-18 (pretrained) → 512-D image feature
- **Text Input**: Short text metadata (tokenized with subword units, e.g., BPE)
- **Text Encoder**: Transformer encoder (2–4 layers, 4–8 heads) → 512-D text embedding
- **Fusion**: Concatenate [512-D image, 512-D text] → 1024-D
- **Dropout**: p=0.3 (randomly drops ~30% of fused features during training)
- **Head**: Linear (1024 → 7), Softmax for probabilities
- **Loss**: Cross-Entropy

## Key Differences

The main difference between the two architectures is the text encoder:
- **Architecture A** uses a GRU (Gated Recurrent Unit) for text processing
- **Architecture B** uses a Transformer encoder with multi-head attention

Architecture B also has a lower dropout rate (0.3 vs 0.5) and uses more sophisticated text tokenization.

## How to Read the Code

### Notebook Structure
Each notebook follows a similar structure:

1. **Architecture Description** (Cell 0): High-level overview of the model architecture
2. **Imports** (Cells 1-2): All necessary Python libraries and dependencies
3. **Data Loading & Preprocessing**: Code for loading and preparing the dataset
4. **Model Definition**: The neural network architecture implementation
5. **Training Loop**: Code for training the model
6. **Evaluation**: Code for testing and evaluating model performance
7. **Results & Analysis**: Visualization and analysis of results

### Key Components to Look For

#### Model Definition
- Look for classes that inherit from `nn.Module`
- The main model class will contain:
  - Image encoder (ResNet-18)
  - Text encoder (GRU or Transformer)
  - Fusion layer (concatenation)
  - Classification head

#### Data Processing
- Image preprocessing: Resizing, normalization, augmentation
- Text preprocessing: Tokenization, padding, vocabulary creation
- Dataset class implementation for handling image-text pairs

#### Training Configuration
- Optimizer settings (learning rate, weight decay)
- Loss function (Cross-Entropy)
- Training parameters (epochs, batch size)
- Validation and testing procedures

### Running the Code

1. **Prerequisites**: Ensure you have the required dependencies installed (see imports in each notebook)
2. **Data**: Make sure the emotion dataset is available and properly formatted
3. **Execution**: Run cells sequentially from top to bottom
4. **Hardware**: These models may require GPU for efficient training

### Expected Outputs

Each notebook should produce:
- Training/validation loss curves
- Accuracy metrics
- Confusion matrices
- Sample predictions on test data
- Model performance comparisons

## Notes

- Both models are designed for 7-class emotion classification
- The architecture descriptions are provided at the top of each notebook
- Code is well-commented and follows PyTorch best practices
- Models use transfer learning with pretrained ResNet-18 for image features
