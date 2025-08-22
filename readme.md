# DDPM Diffusion Model - MNIST Digit Generation

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) for conditional MNIST digit generation. This project demonstrates how diffusion models can learn to generate high-quality images by gradually denoising random noise.

## 🎯 Overview

This implementation includes:

- **Custom U-Net architecture** with time and class conditioning
- **Forward diffusion process** that gradually adds noise to images
- **Reverse diffusion process** that learns to denoise and generate new images
- **Conditional generation** - generate specific digits (0-9)
- **GIF visualization** of the complete denoising process

## 🔥 Generated Results

The model learns to transform pure noise into recognizable MNIST digits. Below are GIFs showing the complete diffusion process for each digit class:

### Digit 0

![Diffusion Process - Digit 0](GIFs/diffusion_process_class_0_epoch_10.gif)

### Digit 1

![Diffusion Process - Digit 1](GIFs/diffusion_process_class_1_epoch_10.gif)

### Digit 2

![Diffusion Process - Digit 2](GIFs/diffusion_process_class_2_epoch_10.gif)

### Digit 3

![Diffusion Process - Digit 3](GIFs/diffusion_process_class_3_epoch_10.gif)

### Digit 4

![Diffusion Process - Digit 4](GIFs/diffusion_process_class_4_epoch_10.gif)

### Digit 5

![Diffusion Process - Digit 5](GIFs/diffusion_process_class_5_epoch_10.gif)

### Digit 6

![Diffusion Process - Digit 6](GIFs/diffusion_process_class_6_epoch_10.gif)

### Digit 7

![Diffusion Process - Digit 7](GIFs/diffusion_process_class_7_epoch_10.gif)

### Digit 8

![Diffusion Process - Digit 8](GIFs/diffusion_process_class_8_epoch_10.gif)

### Digit 9

![Diffusion Process - Digit 9](GIFs/diffusion_process_class_9_epoch_10.gif)

## 🏗️ Architecture

### U-Net Model (`GabiDiffUnet`)

The core model is a U-Net architecture with the following components:

#### **Time and Label Embeddings**

```python
# Sinusoidal position embedding for timesteps
self.time_mlp = nn.Sequential(
    SinusodialPositionEmbedding(time_emb_dim),
    nn.Linear(time_emb_dim, time_emb_dim * 4),
    nn.GELU(),
    nn.Linear(time_emb_dim * 4, time_emb_dim)
)

# Learnable embedding for digit classes (0-9)
self.label_emb = nn.Embedding(num_classes, time_emb_dim)
```

#### **Encoder (Downsampling Path)**

- **4 DownBlocks** with progressively increasing channels (64 → 128 → 256 → 512)
- Each block contains:
  - 2 ResNet blocks with time/label conditioning
  - Attention mechanism (applied to even-indexed layers)
  - Space-to-depth downsampling

#### **Bottleneck**

- 2 ResNet blocks + 1 Attention block
- Processes the most compressed representation

#### **Decoder (Upsampling Path)**

- **4 UpBlocks** with skip connections from encoder
- Each block contains:
  - Transpose convolution for upsampling
  - Concatenation with skip connection
  - 2 ResNet blocks with conditioning
  - Attention mechanism

#### **Key Features**

- **Weight Standardized Convolutions**: Improves training stability
- **Group Normalization**: Better than BatchNorm for small batches
- **SiLU Activation**: Smooth, differentiable activation function
- **Residual Connections**: Helps with gradient flow

## 🔬 Diffusion Process

### Forward Process (Adding Noise)

The forward process gradually corrupts images with Gaussian noise:

```python
# At timestep t, add noise according to:
x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
```

Where:

- `x_0` is the original image
- `α̅_t` is the cumulative product of noise schedule
- `ε` is Gaussian noise

### Reverse Process (Denoising)

The model learns to reverse this process by predicting the noise:

```python
# Model predicts noise ε_θ(x_t, t, class)
# Then we can recover x_{t-1} using:
x_{t-1} = (1/√α_t) * (x_t - (β_t/√(1-α̅_t)) * ε_θ(x_t, t))
```

### Noise Schedules

Two noise schedules are implemented:

1. **Linear Schedule** (used in training):

   ```python
   β_t = linear_interpolation(1e-4, 0.02, num_timesteps)
   ```

2. **Cosine Schedule** (alternative):
   ```python
   α̅_t = cos²((t/T + s)/(1 + s) * π/2)
   ```

## 🎮 Training Process

### Loss Function

The model is trained to predict the noise added at each timestep:

```python
def compute_loss(model, x0, t, labels=None, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    x_t = sample_q(x0, t, noise)  # Add noise
    predicted_noise = model(x_t, t, labels)  # Predict noise
    loss = F.l1_loss(noise, predicted_noise)  # L1 loss
    return loss
```

### Training Loop

1. **Sample batch** of images and labels
2. **Sample random timesteps** t for each image
3. **Add noise** according to forward process
4. **Predict noise** using the model
5. **Compute L1 loss** between actual and predicted noise
6. **Backpropagate** and update model weights

### Conditional Training

The model learns to generate specific digits by conditioning on class labels:

- Label embeddings are added to time embeddings
- This allows controlled generation: "Generate a digit 7"

## 🚀 Sampling (Generation)

### Algorithm

1. **Start with pure noise**: `x_T ~ N(0, I)`
2. **Iteratively denoise** for T steps:
   ```python
   for t in range(T, 0, -1):
       x_{t-1} = sample_p(model, x_t, t, labels)
   ```
3. **Final result**: Clean image x_0

### Features

- **Conditional sampling**: Generate specific digit classes
- **DDIM sampling**: Faster sampling with fewer steps (not implemented yet)
- **Classifier-free guidance**: Could be added for better conditional generation

## 📊 Technical Details

### Model Parameters

- **Time embedding dimension**: 128
- **ResNet depth**: 4 layers
- **Image size**: 32×32 (upscaled from 28×28 MNIST)
- **Input channels**: 1 (grayscale MNIST)
- **Number of classes**: 10 (digits 0-9)
- **Timesteps**: 1000

### Training Hyperparameters

- **Learning rate**: 1e-4
- **Batch size**: 64
- **Optimizer**: Adam
- **Loss function**: L1 (mean absolute error)
- **Epochs**: 1000

### Memory and Performance

- **Model size**: ~50M parameters
- **Training time**: ~hours on GPU
- **Inference time**: ~30 seconds per batch (1000 steps)

## 📁 Project Structure

```
DDPM-diffusion/
├── custom_diffusion_model_experiments.ipynb  # Main development notebook
├── custom_diffusion_model_training.py        # Standalone training script
├── generating_gif.ipynb                      # GIF generation code
├── saved_model.pth                          # Trained model weights
├── data/MNIST/                              # MNIST dataset
├── GIFs/                                    # Generated diffusion GIFs
├── results/                                 # Training samples
└── requirements.txt                         # Dependencies
```

## 🔧 Key Implementation Features

### Custom Modules

1. **SpaceToDepth**: Efficient downsampling using channel dimension
2. **WeightStandardizedConv2d**: Normalized convolutions for stability
3. **SinusoidalPositionEmbedding**: Time encoding for diffusion steps
4. **ResnetBlock**: Residual blocks with time/label conditioning
5. **Attention**: Self-attention for capturing long-range dependencies

### Advanced Techniques

- **Gradient clipping**: Prevents exploding gradients
- **Exponential moving averages**: Smoother model updates (could be added)
- **Progressive training**: Start with fewer timesteps (could be implemented)

## 🎨 Visualization

The project includes comprehensive visualization:

- **Training samples**: Saved every 1000 batches
- **Diffusion GIFs**: Complete denoising process visualization
- **Loss tracking**: Monitor training progress
- **Conditional samples**: Generate specific digit classes

## 🚀 Future Improvements

1. **DDIM Sampling**: Faster inference with deterministic sampling
2. **Classifier-free Guidance**: Better conditional generation
3. **Progressive Training**: Start with fewer timesteps
4. **FID/IS Metrics**: Quantitative evaluation
5. **Higher Resolution**: Scale to larger images
6. **Other Datasets**: CIFAR-10, CelebA, etc.

## 📚 References

- **DDPM Paper**: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- **Improved DDPM**: "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
- **DDIM**: "Denoising Diffusion Implicit Models" (Song et al., 2020)

## 🏃‍♂️ Quick Start

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run training**:

   ```python
   python custom_diffusion_model_training.py
   ```

3. **Generate samples**:

   ```python
   # Load trained model and sample
   model.eval()
   samples = sampling(model, (10, 1, 32, 32), labels=torch.arange(10))
   ```

4. **Create GIFs**:
   ```python
   # Run generating_gif.ipynb to create visualization GIFs
   ```

This implementation demonstrates the power of diffusion models for high-quality image generation with the added benefit of conditional control over the generated content.
