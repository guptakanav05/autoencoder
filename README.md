# Image Compression and Reconstruction using Autoencoders

## Overview

This project implements **autoencoders** for image compression and reconstruction using PyTorch on the **EMNIST (Extended MNIST)** dataset. The EMNIST dataset contains 28×28 grayscale images of handwritten letters (A-Z).

An autoencoder is an unsupervised neural network that learns to compress (encode) data into a lower-dimensional representation and then reconstruct (decode) it back to the original form.

## What is an Autoencoder?

```
Input Image → [ENCODER] → Latent Vector (bottleneck) → [DECODER] → Reconstructed Image
(28×28)                     (32 or 64 dims)                          (28×28)
```

### Key Components:
1. **Encoder**: Compresses the input image into a compact latent representation
2. **Bottleneck (Latent Space)**: The compressed representation — smaller size means more compression
3. **Decoder**: Reconstructs the image from the latent representation
4. **Reconstruction Loss**: We train the model to minimize MSE (Mean Squared Error) between input and output

### Why Autoencoders?
- **Dimensionality reduction**: Compress 784 pixels → 32 or 64 numbers
- **Feature learning**: The latent space captures meaningful patterns
- **Data denoising**: A trained autoencoder can remove noise from images
- **Generative models**: The foundation for Variational Autoencoders (VAEs)

## Model Architectures

We implement **4 different architectures** to compare performance:

### Fully Connected (ANN) Autoencoders:

| Model | Architecture | Bottleneck | Key Feature |
|-------|-------------|------------|-------------|
| FC v1 | 784→256→128→**32**→128→256→784 | 32 | Simple, fewer parameters |
| FC v2 | 784→512→256→128→**64**→128→256→512→784 | 64 | Deeper + BatchNorm |

### Convolutional (CNN) Autoencoders:

| Model | Architecture | Bottleneck | Key Feature |
|-------|-------------|------------|-------------|
| CNN v1 | Conv(1→16→32)→FC→**32**→FC→ConvT(32→16→1) | 32 | Spatial feature extraction |
| CNN v2 | Conv(1→32→64→128)→FC→**64**→FC→ConvT(128→64→32→1) | 64 | Deeper + BatchNorm |

### Why CNN > FC for images?
- CNNs preserve **spatial structure** (pixel neighborhoods matter)
- **Weight sharing** across the image (fewer parameters)
- **Translation invariance** (features detected anywhere in the image)
- **Local receptive fields** (each neuron looks at a small patch)

## Concepts Used

| Concept | Where Used |
|---------|-----------|
| Linear (Dense) layers | FC autoencoders |
| Conv2d / ConvTranspose2d | CNN autoencoders |
| ReLU activation | All models (hidden layers) |
| Sigmoid activation | All models (output layer, maps to [0,1]) |
| Batch Normalization | FC v2, CNN v2 (stabilizes training) |
| MaxPool2d | CNN encoders (downsampling) |
| MSE Loss | Reconstruction loss function |
| Adam Optimizer | Weight updates |
| Train/Val Split | Monitor overfitting |
| t-SNE | Latent space visualization |

## Project Structure

```
autoencoder_project/
├── data_loader.py      # Dataset loading & preprocessing
├── models.py           # All 4 autoencoder architectures
├── train.py            # Training loop with loss tracking
├── visualize.py        # Visualization utilities
├── main.py             # Main script (run this!)
├── requirements.txt    # Dependencies
├── README.md           # This file
├── data/               # Auto-downloaded dataset (created on first run)
└── results/            # Saved plots and model checkpoints
```

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Project
```bash
python main.py
```

This will:
1. Download the EMNIST dataset (first run only, ~500MB)
2. Train all 4 autoencoder models (20 epochs each)
3. Generate all visualizations in the `results/` folder
4. Print a summary comparison table

### 3. Check Results
After running, check the `results/` folder for:
- `{model}_reconstructions.png` — Input vs reconstructed images
- `{model}_loss_curves.png` — Training and validation loss
- `{model}_latent_space.png` — t-SNE visualization of latent space
- `all_models_loss_comparison.png` — Side-by-side loss comparison

## Expected Observations

1. **CNN models should achieve lower loss** than FC models because they better capture spatial patterns in images
2. **Larger bottleneck (64) should give better reconstruction** than smaller (32), but less compression
3. **BatchNorm models (v2)** should train faster and more stably
4. **Latent space clusters** should be more distinct for CNN models (better feature learning)
5. **Trade-off**: Smaller bottleneck = more compression but blurrier reconstructions

## Dataset

- **EMNIST Letters Split**: Handwritten A-Z characters
- **Image size**: 28×28 pixels, grayscale (1 channel)
- **Train set**: ~88,800 images (after 80/20 split: ~71,040 train, ~17,760 validation)
- **Test set**: ~14,800 images
- **Auto-downloaded** via PyTorch's `torchvision.datasets.EMNIST`
