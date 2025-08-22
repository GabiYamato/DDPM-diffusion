import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import numpy as np
import seaborn as sns
from PIL import Image
import os
import time
import logging
import random
import math
from typing import Optional, Tuple, List, Dict, Any
import warnings
from pathlib import Path
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

# For reproducibility
warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Helper functions for the model
def space_to_depth(x, size=2):
    """
    Downscale method that uses the depth dimension to
    downscale the spatial dimensions
    """
    b, c, h, w = x.shape
    assert h % size == 0 and w % size == 0, "height/width must be divisible by size"
    out_h = h // size
    out_w = w // size
    out_c = c * (size * size)

    x = x.reshape((b, c, out_h, size, out_w, size))
    x = x.permute((0, 1, 3, 5, 2, 4))
    x = x.reshape((b, out_c, out_h, out_w))
    return x


class SpaceToDepth(nn.Module):
    def __init__(self, size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size

    def forward(self, x):
        return space_to_depth(x, self.size)


class Residual(nn.Module):
    """
    Apply residual connection using an input function
    """
    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = func

    def forward(self, x, *args, **kwargs):
        return x + self.func(x, *args, **kwargs)


def upsample(in_channels, out_channels=None):
    out_channels = in_channels if out_channels is None else out_channels
    seq = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_channels, out_channels, 3, padding=1)
    )
    return seq


def downsample(in_channels, out_channels=None):
    out_channels = in_channels if out_channels is None else out_channels
    seq = nn.Sequential(
        SpaceToDepth(2),
        nn.Conv2d(4 * in_channels, out_channels, 1)
    )
    return seq


class SinusodialPositionEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, time_steps):
        positions = torch.unsqueeze(time_steps, 1)
        embeddings = torch.zeros((time_steps.shape[0], self.embedding_dim), device=time_steps.device)
        denominators = 10_000 ** (2 * torch.arange(self.embedding_dim // 2, device=time_steps.device) / self.embedding_dim)
        embeddings[:, 0::2] = torch.sin(positions / denominators)
        embeddings[:, 1::2] = torch.cos(positions / denominators)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    """
    Weight Standardized Conv2d compatible with nn.Conv2d args.
    https://arxiv.org/abs/1903.10520
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        variance = weight.var(dim=[1, 2, 3], keepdim=True, correction=0)
        normalized_weight = (weight - mean) / torch.sqrt(variance + eps)
        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim=None, groups=8):
        super().__init__()
        if time_embed_dim is not None:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, 2 * out_channels)
            )
        else:
            self.mlp = None

        self.block1 = Block(in_channels, out_channels, groups)
        self.block2 = Block(out_channels, out_channels, groups)

        if in_channels == out_channels:
            self.res_conv = nn.Identity()
        else:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_embedding=None):
        scale_shift = None
        if self.mlp is not None and time_embedding is not None:
            time_emb = self.mlp(time_embedding)
            time_emb = time_emb.view(*time_emb.shape, 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, in_channels, num_heads=4, dim_head=32):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale_factor = 1 / (dim_head) ** 0.5
        self.hidden_dim = num_heads * dim_head
        self.input_to_qkv = nn.Conv2d(in_channels, 3 * self.hidden_dim, 1, bias=False)
        self.to_output = nn.Conv2d(self.hidden_dim, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.input_to_qkv(x)
        q, k, v = map(lambda t: t.view(b, self.num_heads, self.dim_head, h * w), qkv.chunk(3, dim=1))
        q = q * self.scale_factor
        sim = torch.einsum("b h c i, b h c j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attention = sim.softmax(dim=-1)
        output = torch.einsum("b h i j, b h c j -> b h i c", attention, v)
        output = output.permute(0, 1, 3, 2).reshape((b, self.hidden_dim, h, w))
        return self.to_output(output)


class LinearAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, dim_head=32):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale_factor = 1 / (dim_head) ** 0.5
        self.hidden_dim = num_heads * dim_head
        self.input_to_qkv = nn.Conv2d(in_channels, 3 * self.hidden_dim, 1, bias=False)
        self.to_output = nn.Sequential(
            nn.Conv2d(self.hidden_dim, in_channels, 1),
            nn.GroupNorm(1, in_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.input_to_qkv(x)
        q, k, v = map(lambda t: t.view(b, self.num_heads, self.dim_head, h * w), qkv.chunk(3, dim=1))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale_factor
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        output = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        output = output.view((b, self.hidden_dim, h, w))
        return self.to_output(output)


class PreGroupNorm(nn.Module):
    def __init__(self, dim, func, groups=1):
        super().__init__()
        self.func = func
        self.group_norm = nn.GroupNorm(groups, dim)

    def forward(self, x):
        x = self.group_norm(x)
        x = self.func(x)
        return x


# U-Net Model with Conditioning
debug_mode = False

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_attn=False):
        super().__init__()
        self.block1 = ResnetBlock(in_ch, out_ch, time_embed_dim=time_emb_dim)
        self.block2 = ResnetBlock(out_ch, out_ch, time_embed_dim=time_emb_dim)
        self.attn = Attention(out_ch) if use_attn else nn.Identity()
        self.down = downsample(out_ch)

    def forward(self, x, cond):
        if debug_mode:
            print(f"DownBlock input: {x.shape}, cond: {getattr(cond,'shape',None)}")
        x = self.block1(x, cond)
        if debug_mode:
            print(f" after block1: {x.shape}")
        x = self.block2(x, cond)
        if debug_mode:
            print(f" after block2: {x.shape}")
        x = self.attn(x)
        if debug_mode:
            print(f" after attn: {x.shape}")

        # store this for skip connection
        skip = x  
        
        x = self.down(x)
        if debug_mode:
            print(f" after downsample: {x.shape}")
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim, use_attn=False):
        super().__init__()
        # First upsample the input to match skip connection size
        self.upconv = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1)
        # Then process concatenated features
        self.block1 = ResnetBlock(in_ch + skip_ch, out_ch, time_embed_dim=time_emb_dim)
        self.block2 = ResnetBlock(out_ch, out_ch, time_embed_dim=time_emb_dim)
        self.attn = Attention(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, cond):
        if debug_mode:
            print(f"upblock input: {x.shape}, skip: {getattr(skip,'shape',None)}, cond: {getattr(cond,'shape',None)} ")

        # Upsample to match skip connection spatial dimensions
        x = self.upconv(x)
        if debug_mode:
            print(f" after upconv: {x.shape}")

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        if debug_mode:
            print(f" after concat: {x.shape}")

        # Process with resnet blocks
        x = self.block1(x, cond)
        if debug_mode:
            print(f" after block1: {x.shape}")

        x = self.block2(x, cond)
        if debug_mode:
            print(f" after block2: {x.shape}")

        x = self.attn(x)
        if debug_mode:
            print(f" after attn: {x.shape}")
        
        return x


class GabiDiffUnet(nn.Module):

    def __init__(self,time_emb_dim=128, num_classes = 10, in_channels = 3,resnet_depth=4, image_size=28):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        #time and label embedding
        self.time_mlp = nn.Sequential(
            SinusodialPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        #DOWNSAMPLE 
        self.downsample = nn.ModuleList()
        for i in range(resnet_depth):
            in_ch = in_channels if i == 0 else out_ch
            out_ch = 64 * (2 ** i)
            self.downsample.append(
                DownBlock(in_ch, out_ch, time_emb_dim=time_emb_dim, use_attn=(i % 2 == 0))
            )

        #BOTTLENECK
        self.bottleneck = nn.Sequential(
            ResnetBlock(64 * (2 ** (resnet_depth - 1)), 64 * (2 ** (resnet_depth - 1)), time_embed_dim=time_emb_dim),
            ResnetBlock(64 * (2 ** (resnet_depth - 1)), 64 * (2 ** (resnet_depth - 1)), time_embed_dim=time_emb_dim),
            Attention(64 * (2 ** (resnet_depth - 1)))
        )

        self.upsample = nn.ModuleList()
        ch_list = [64 * (2 ** i) for i in range(resnet_depth)]  # [64, 128, 256, 512]

        for i in range(resnet_depth - 1, -1, -1):
            in_ch = ch_list[i] if i == resnet_depth - 1 else out_ch  # coming from below
            skip_ch = ch_list[i]  # skip connection channels
            out_ch = skip_ch if i > 0 else 64  # final goes back to 64
            self.upsample.append(
                UpBlock(in_ch, skip_ch, out_ch, time_emb_dim=time_emb_dim, use_attn=(i % 2 == 0))
            )


        #FINAL BLOCK
        self.final_block = nn.Sequential(
            ResnetBlock(64, 64, time_embed_dim=time_emb_dim),
            ResnetBlock(64, 64, time_embed_dim=time_emb_dim),
            nn.Conv2d(64, in_channels, kernel_size=1)
        )

    def forward(self, x, time_stamp, labels=None):

        if debug_mode:
            print(f"Model input x: {x.shape}, time_stamp: {getattr(time_stamp,'shape',None)}, labels: {getattr(labels,'shape',None)}")

        #Time and Label Embeddings
        time_emb = self.time_mlp(time_stamp)
        if debug_mode:
            print(f" time_emb: {time_emb.shape}")
        label_emb = self.label_emb(labels) if labels is not None else torch.zeros_like(time_emb)
        
        if debug_mode:
            print(f" label_emb: {getattr(label_emb,'shape',None)}")

        cond = time_emb + label_emb
        if debug_mode:
            print(f" cond: {cond.shape}")

        skips = []
        # down path
        for i, layer in enumerate(self.downsample):
            x, skip = layer(x, cond)
            skips.append(skip)
            if debug_mode:
                print(f" after downsample layer {i}: {x.shape}, skip saved: {skip.shape}")



        #bottleneck
        x = self.bottleneck[0](x, cond)
        if debug_mode:
            print(f" after bottleneck block0: {x.shape}")
        x = self.bottleneck[1](x, cond)
        if debug_mode:
            print(f" after bottleneck block1: {x.shape}")
        x = self.bottleneck[2](x)
        if debug_mode:
            print(f" after bottleneck attn: {x.shape}")

        #up path
        for i, layer in enumerate(self.upsample):

            skip = skips[-(i+1)]
            if debug_mode:
                print(f"skips shape: {skip.shape}")
            x = layer(x, skip, cond)
            if debug_mode:
                print(f" after upsample layer {i}: {x.shape}")

        h0 = self.final_block[0](x, cond)
        if debug_mode:
            print(f" final_block[0] out: {h0.shape}")
        h1 = self.final_block[1](h0, cond)
        if debug_mode:
            print(f" final_block[1] out: {h1.shape}")
        k = self.final_block[2](h1)
        if debug_mode:
            print(f" final output k: {k.shape}")

        return k 


# Diffusion Process Functions
def linear_schedule(num_timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    return betas

def cosine_schedule(num_timesteps, s=0.008):
    def f(t):
        return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2
    x = torch.linspace(0, num_timesteps, num_timesteps + 1)
    alphas_cumprod = f(x) / f(torch.tensor([0]))
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = torch.clip(betas, 0.0001, 0.999)
    return betas

def sample_by_t(tensor_to_sample, timesteps, x_shape):
    batch_size = timesteps.shape[0]
    sampled_tensor = tensor_to_sample.gather(-1, timesteps.cpu())
    sampled_tensor = torch.reshape(sampled_tensor, (batch_size,) + (1,) * (len(x_shape) - 1))
    return sampled_tensor.to(timesteps.device)

# Initialize diffusion parameters
num_timesteps = 1001
betas_t = linear_schedule(num_timesteps)

alphas_t = 1-betas_t
sqrt_alphas_t = torch.sqrt(alphas_t)
one_over_sqrt_alphas_t = 1. / sqrt_alphas_t

alphas_bar_t = torch.cumprod(alphas_t,dim=0)
alphas_bar_t_minus_1 = torch.cat((torch.tensor([0]),alphas_bar_t[:-1]))
sqrt_alphas_bar_t_minus_1 = torch.sqrt(alphas_bar_t_minus_1)

sqrt_alphas_bar_t = torch.sqrt(alphas_bar_t)
sqrt_1_minus_alphas_bar_t = torch.sqrt(1. - alphas_bar_t)

posterior_variance = (1. -alphas_bar_t_minus_1) / (1. - alphas_bar_t) * betas_t

def sample_q(x0, t, noise=None):
    """Forward sampling process - add noise to image"""
    if noise is None:
        noise = torch.randn_like(x0)
        
    sqrt_alphas_bar_t_sampled = sample_by_t(sqrt_alphas_bar_t, t, x0.shape)
    sqrt_1_minus_alphas_bar_t_sampled = sample_by_t(sqrt_1_minus_alphas_bar_t, t, x0.shape)
    
    x_t = sqrt_alphas_bar_t_sampled * x0 + sqrt_1_minus_alphas_bar_t_sampled * noise
    return x_t

@torch.no_grad()
def sample_p(model, x_t, t, labels=None, clipping=True):
    """
    Sample from p_θ(xₜ₋₁|xₜ) to get xₜ₋₁ according to Algorithm 2
    """
    betas_t_sampled = sample_by_t(betas_t, t, x_t.shape)
    sqrt_1_minus_alphas_bar_t_sampled = sample_by_t(sqrt_1_minus_alphas_bar_t, t, x_t.shape)
    one_over_sqrt_alphas_t_sampled = sample_by_t(one_over_sqrt_alphas_t, t, x_t.shape)

    if clipping:
        sqrt_alphas_bar_t_sampled = sample_by_t(sqrt_alphas_bar_t, t, x_t.shape)
        sqrt_alphas_bar_t_minus_1_sampled = sample_by_t(sqrt_alphas_bar_t_minus_1, t, x_t.shape)
        alphas_bar_t_sampled = sample_by_t(alphas_bar_t, t, x_t.shape)
        sqrt_alphas_t_sampled = sample_by_t(sqrt_alphas_t, t, x_t.shape)
        alphas_bar_t_minus_1_sampled = sample_by_t(alphas_bar_t_minus_1, t, x_t.shape)

        x0_reconstruct = 1 / sqrt_alphas_bar_t_sampled * (x_t - sqrt_1_minus_alphas_bar_t_sampled * model(x_t, t, labels))
        x0_reconstruct = torch.clip(x0_reconstruct, -1., 1.)
        predicted_mean = (sqrt_alphas_bar_t_minus_1_sampled * betas_t_sampled) / (1 - alphas_bar_t_sampled) * x0_reconstruct + (sqrt_alphas_t_sampled * (1 - alphas_bar_t_minus_1_sampled)) /  (1 - alphas_bar_t_sampled) * x_t

    else:
        predicted_mean = one_over_sqrt_alphas_t_sampled * (x_t - betas_t_sampled / sqrt_1_minus_alphas_bar_t_sampled * model(x_t, t, labels))

    if t[0].item() == 1:
        return predicted_mean
    else:
        posterior_variance_sampled = sample_by_t(posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return predicted_mean + torch.sqrt(posterior_variance_sampled) * noise

@torch.no_grad()
def sampling(model, shape, labels=None, image_noise_steps_to_keep=1):
    """
    Implementing Algorithm 2 - sampling.
    Args:
        model (torch.Module): the model that predicts the noise
        shape (tuple): shape of the data (batch, channels, image_size, image_size)
        labels (torch.Tensor, optional): class labels for conditional generation
    Returns:
        (list): list containing the images in the different steps of the reverse process
    """

    batch = shape[0]
    images = torch.randn(shape, device=device)  # pure noise
    images_list = []

    # Fix: Change range to start from num_timesteps-1 instead of num_timesteps
    for timestep in tqdm(range(num_timesteps-1, 0, -1), desc='sampling timestep'):
        images = sample_p(model, images, torch.full((batch,), timestep, device=device, dtype=torch.long), labels)
        if timestep <= image_noise_steps_to_keep:
            images_list.append(images.cpu())
    return images_list

# Training functions
def compute_loss(model, x0, t, labels=None, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    
    x_t = sample_q(x0, t, noise)
    predicted_noise = model(x_t, t, labels)
    loss = F.l1_loss(noise, predicted_noise)
    return loss

def sample_t(batch_size, num_timesteps=num_timesteps, device=device):
    # returns uint64/long tensor with values in [0, num_timesteps-1]
    return torch.randint(0, num_timesteps, (batch_size,), device=device, dtype=torch.long)

def save_checkpoint(state, filepath):
    torch.save(state, filepath)

def format_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def main():
    # Data loading
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Hyperparameters
    parameters = {
        "time_embed_dim": 128,
        "resnet_depth": 4,
        "learning_rate": 1e-4,
        "batch_size": 64,
        "num_epochs": 1000
    }

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=parameters["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=parameters["batch_size"], shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Model initialization
    model = GabiDiffUnet(
        time_emb_dim=parameters["time_embed_dim"],
        num_classes=10,
        resnet_depth=parameters["resnet_depth"],
        image_size=32,
        in_channels=1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=parameters["learning_rate"])

    # Results folder
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)

    # Load saved model if exists
    model_saved_file_path = Path('./saved_model.pth')
    start_epoch = 0
    
    if model_saved_file_path.exists():
        print('Loading saved model...')
        saved_model = torch.load(str(model_saved_file_path), map_location=device)
        model.load_state_dict(saved_model['model'])
        optimizer.load_state_dict(saved_model['optimizer'])
        start_epoch = saved_model['epoch']
        print(f'Resuming from epoch {start_epoch}')
    else:
        print('Starting training from scratch...')

    # Training parameters
    epochs = parameters["num_epochs"]
    log_interval = 5
    sample_every = 1000
    checkpoint_every = 10

    start_time = time.time()

    print(f"Starting training on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_index, (data, labels) in enumerate(train_loader):
            images = data.to(device)
            labels = labels.to(device).long()

            # Sample random timesteps
            t = sample_t(images.shape[0]).to(device).long()

            # Compute loss
            loss = compute_loss(model, images, t, labels)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Log progress
            if batch_index % log_interval == 0:
                current_time = time.time()
                elapsed_time = current_time - start_time
                batches_done = epoch * len(train_loader) + batch_index + 1
                total_batches = epochs * len(train_loader)
                remaining_time = (elapsed_time / batches_done) * (total_batches - batches_done)
                
                print(f"Epoch [{epoch}/{epochs}], Batch [{batch_index}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, ETA: {format_time(remaining_time)}")

            # Sample images periodically
            if batch_index % sample_every == 0:
                print("Generating samples...")
                model.eval()
                with torch.no_grad():
                    # Sample images conditionally using labels
                    sample_labels = torch.arange(10, dtype=torch.long, device=device)
                    sampled_images = sampling(model, (10, 1, 32, 32), labels=sample_labels, image_noise_steps_to_keep=999)
                    
                    # Save the sampled images
                    for timestep_idx, img in enumerate(sampled_images):
                        for class_idx in range(10):
                            single_img = img[class_idx:class_idx+1]
                            save_image(single_img, 
                                     results_folder / f"sampled_epoch{epoch}_batch{batch_index}_timestep{timestep_idx}_class{class_idx}.png", 
                                     normalize=True)
                model.train()

            # Save checkpoint
            if batch_index % checkpoint_every == 0:
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'batch_index': batch_index
                }, model_saved_file_path)

        # End of epoch
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

        # Save model at end of each epoch
        save_checkpoint({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'batch_index': 0
        }, model_saved_file_path)

    print("Training completed!")

if __name__ == "__main__":
    main()