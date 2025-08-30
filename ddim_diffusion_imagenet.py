"""
DDIM Text-to-Image Diffusion Model for ImageNet
===============================================

A complete implementation of DDIM (Denoising Diffusion Implicit Models) for text-to-image generation
using CLIP text embeddings and ImageNet data. DDIM enables much faster sampling (10-50 steps vs 1000).

Features:
- DDIM sampling for fast generation
- CLIP text conditioning
- Classifier-free guidance
- Complete training and inference pipeline
- Interactive demo with step visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

import torchvision
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from torchvision.models import inception_v3

import clip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import warnings
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import linalg
import math

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# HELPER FUNCTIONS AND MODULES
# =============================================================================

def space_to_depth(x, size=2):
    """Downscale method using depth dimension"""
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

class SinusoidalPositionEmbedding(nn.Module):
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
    """Weight Standardized Conv2d for improved training stability"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        variance = weight.var(dim=[1, 2, 3], keepdim=True, correction=0)
        normalized_weight = (weight - mean) / torch.sqrt(variance + eps)
        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

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

class CrossAttention(nn.Module):
    """Cross attention between image features and text embeddings"""
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * num_heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        # Reshape spatial dimensions to sequence
        x_seq = x.view(b, c, h * w).transpose(1, 2)  # (b, h*w, c)
        
        q = self.to_q(x_seq)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = q.view(b, h * w, self.num_heads, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.dim_head).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.dim_head).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, h * w, -1)
        out = self.to_out(out)
        
        # Reshape back to spatial
        out = out.transpose(1, 2).view(b, c, h, w)
        
        return out + x  # Residual connection

# =============================================================================
# TEXT CONDITIONING MODULES
# =============================================================================

class TextEmbeddingProjector(nn.Module):
    """Projects CLIP text embeddings to model dimension"""
    def __init__(self, text_embed_dim: int = 512, model_embed_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(text_embed_dim, model_embed_dim * 2),
            nn.GELU(),
            nn.Linear(model_embed_dim * 2, model_embed_dim),
            nn.LayerNorm(model_embed_dim)
        )
    
    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection(text_embeddings)

class TextConditionalResnetBlock(ResnetBlock):
    """ResNet block with text conditioning via cross-attention"""
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int,
                 text_embed_dim: int, groups: int = 8, use_cross_attn: bool = True):
        super().__init__(in_channels, out_channels, time_embed_dim, groups)
        
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = CrossAttention(
                query_dim=out_channels,
                context_dim=text_embed_dim,
                num_heads=4,
                dim_head=out_channels // 4
            )
    
    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor] = None,
                text_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Standard ResNet forward pass
        x = super().forward(x, time_embedding)
        
        # Apply cross-attention with text
        if self.use_cross_attn and text_embedding is not None:
            if text_embedding.dim() == 2:
                text_embedding = text_embedding.unsqueeze(1)  # Add sequence dimension
            x = self.cross_attn(x, text_embedding)
        
        return x

# =============================================================================
# U-NET ARCHITECTURE
# =============================================================================

class DownBlock(nn.Module):
    """Down block with text conditioning"""
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, text_emb_dim: int, use_attn: bool = False):
        super().__init__()
        self.block1 = TextConditionalResnetBlock(in_ch, out_ch, time_emb_dim, text_emb_dim, use_cross_attn=use_attn)
        self.block2 = TextConditionalResnetBlock(out_ch, out_ch, time_emb_dim, text_emb_dim, use_cross_attn=use_attn)
        self.attn = Attention(out_ch) if use_attn else nn.Identity()
        
        # Downsample using space-to-depth
        self.down = nn.Sequential(
            SpaceToDepth(2),
            nn.Conv2d(4 * out_ch, out_ch, 1)
        )

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor, text_cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.block1(x, time_cond, text_cond)
        x = self.block2(x, time_cond, text_cond)
        x = self.attn(x)
        
        skip = x
        x = self.down(x)
        
        return x, skip

class UpBlock(nn.Module):
    """Up block with text conditioning"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_emb_dim: int, text_emb_dim: int, use_attn: bool = False):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1)
        self.block1 = TextConditionalResnetBlock(in_ch + skip_ch, out_ch, time_emb_dim, text_emb_dim, use_cross_attn=use_attn)
        self.block2 = TextConditionalResnetBlock(out_ch, out_ch, time_emb_dim, text_emb_dim, use_cross_attn=use_attn)
        self.attn = Attention(out_ch) if use_attn else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_cond: torch.Tensor, text_cond: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, time_cond, text_cond)
        x = self.block2(x, time_cond, text_cond)
        x = self.attn(x)
        return x

class TextConditionalUNet(nn.Module):
    """U-Net with text conditioning for text-to-image diffusion"""
    
    def __init__(self, text_embed_dim: int = 512, model_channels: int = 128, 
                 resnet_depth: int = 4, image_size: int = 256, in_channels: int = 3):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        
        # Time embedding
        time_emb_dim = model_channels
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Text embedding projection
        self.text_projector = TextEmbeddingProjector(text_embed_dim, model_channels)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling path
        self.downsample = nn.ModuleList()
        ch_list = []
        for i in range(resnet_depth):
            in_ch = model_channels if i == 0 else out_ch
            out_ch = model_channels * (2 ** i)
            ch_list.append(out_ch)
            
            self.downsample.append(
                DownBlock(
                    in_ch, out_ch, time_emb_dim, model_channels, 
                    use_attn=(i >= 2)  # Use attention in deeper layers
                )
            )
        
        # Bottleneck
        bottleneck_ch = ch_list[-1]
        self.bottleneck = nn.Sequential(
            TextConditionalResnetBlock(bottleneck_ch, bottleneck_ch, time_emb_dim, model_channels, use_cross_attn=True),
            TextConditionalResnetBlock(bottleneck_ch, bottleneck_ch, time_emb_dim, model_channels, use_cross_attn=True),
            Attention(bottleneck_ch)
        )
        
        # Upsampling path
        self.upsample = nn.ModuleList()
        for i in range(resnet_depth - 1, -1, -1):
            in_ch = ch_list[i] if i == resnet_depth - 1 else out_ch
            skip_ch = ch_list[i]
            out_ch = skip_ch if i > 0 else model_channels
            
            self.upsample.append(
                UpBlock(
                    in_ch, skip_ch, out_ch, time_emb_dim, model_channels,
                    use_attn=(i >= 2)
                )
            )
        
        # Final output layers
        self.final_block = nn.Sequential(
            TextConditionalResnetBlock(model_channels, model_channels, time_emb_dim, model_channels, use_cross_attn=True),
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, 
                text_embeddings: torch.Tensor, cfg_scale: float = 1.0) -> torch.Tensor:
        
        # Handle classifier-free guidance during training
        if cfg_scale != 1.0 and self.training:
            # During training, randomly replace some text embeddings with zeros for CFG
            batch_size = text_embeddings.size(0)
            cfg_mask = torch.rand(batch_size, device=text_embeddings.device) < 0.1  # 10% dropout
            text_embeddings = text_embeddings.clone()
            text_embeddings[cfg_mask] = 0
        
        # Time and text embeddings
        time_emb = self.time_mlp(timesteps)
        text_emb = self.text_projector(text_embeddings)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Downsampling
        skips = []
        for layer in self.downsample:
            x, skip = layer(x, time_emb, text_emb)
            skips.append(skip)
        
        # Bottleneck
        for i, layer in enumerate(self.bottleneck):
            if isinstance(layer, TextConditionalResnetBlock):
                x = layer(x, time_emb, text_emb)
            else:
                x = layer(x)
        
        # Upsampling
        for i, layer in enumerate(self.upsample):
            skip = skips[-(i+1)]
            x = layer(x, skip, time_emb, text_emb)
        
        # Final output
        if isinstance(self.final_block[0], TextConditionalResnetBlock):
            x = self.final_block[0](x, time_emb, text_emb)
            x = self.final_block[1:](x)
        else:
            x = self.final_block(x)
        
        return x

# =============================================================================
# DDIM SCHEDULER
# =============================================================================

class DDIMScheduler:
    """DDIM (Denoising Diffusion Implicit Models) Scheduler for fast sampling"""
    
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, 
                 beta_end: float = 0.02, beta_schedule: str = "linear"):
        self.num_train_timesteps = num_train_timesteps
        
        # Create beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # For DDIM sampling
        self.final_alpha_cumprod = 1.0
        
    def _cosine_schedule(self, timesteps, s=0.008):
        """Cosine noise schedule"""
        def f(t):
            return torch.cos((t / timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2
        x = torch.linspace(0, timesteps, timesteps + 1)
        alphas_cumprod = f(x) / f(torch.tensor([0]))
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clip(betas, 0.0001, 0.9999)
    
    def set_timesteps(self, num_inference_steps: int, device: torch.device = None):
        """Set the timesteps for DDIM sampling"""
        self.num_inference_steps = num_inference_steps
        
        # Create inference schedule - evenly spaced timesteps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round().long()
        timesteps = torch.flip(timesteps, dims=[0])  # Reverse for denoising
        
        self.timesteps = timesteps
        if device is not None:
            self.timesteps = self.timesteps.to(device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)
    
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, 
                  timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to samples (forward process)"""
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        
        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor,
             eta: float = 0.0, use_clipped_model_output: bool = False) -> torch.Tensor:
        """
        Perform one DDIM denoising step
        
        Args:
            model_output: Direct output from the learned diffusion model
            timestep: Current discrete timestep in the diffusion chain
            sample: Current instance of sample being created by diffusion process
            eta: Weight of noise for DDIM sampling (0.0 = deterministic, 1.0 = DDPM)
        """
        
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        
        beta_prod_t = 1 - alpha_prod_t
        
        # 3. compute predicted original sample from predicted noise
        if use_clipped_model_output:
            model_output = torch.clamp(model_output, -1, 1)
        
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        
        # 4. Clip predicted x_0
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        variance = self._get_variance(timestep, prev_timestep, eta)
        std_dev_t = variance ** (0.5)
        
        # 6. compute "direction pointing to x_t" of formula (12)
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output
        
        # 7. compute x_t without "random noise" of formula (12)
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        
        if eta > 0:
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + std_dev_t * noise
        
        return prev_sample
    
    def _get_variance(self, timestep, prev_timestep, eta):
        """Compute variance for DDIM sampling"""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        variance = variance * eta ** 2
        
        return variance

# =============================================================================
# DATA LOADING
# =============================================================================

class ImageNetMiniDataset(Dataset):
    """ImageNet Mini dataset with CLIP text embeddings"""
    
    def __init__(self, root_dir: str, clip_model, transform=None, split='train', 
                 max_classes: Optional[int] = None):
        self.root_dir = Path(root_dir)
        self.clip_model = clip_model
        self.transform = transform
        self.split = split
        
        # Load ImageNet class mappings from words.txt
        self.class_to_name = self._load_class_mappings()
        
        # Get available classes from directory structure
        self.available_classes = self._get_available_classes()
        
        # Limit classes if specified
        if max_classes:
            self.available_classes = self.available_classes[:max_classes]
        
        print(f"Found {len(self.available_classes)} classes in {split} split")
        
        # Create dataset samples
        self.samples = []
        self._load_samples()
        
        # Pre-compute CLIP embeddings for all classes
        self.text_embeddings = self._compute_text_embeddings()
        
    def _load_class_mappings(self) -> Dict[str, str]:
        """Load ImageNet class ID to human-readable name mappings from words.txt"""
        words_file = self.root_dir / "words.txt"
        class_mappings = {}
        
        if words_file.exists():
            with open(words_file, 'r') as f:
                for line in f:
                    if '\t' in line:
                        class_id, class_name = line.strip().split('\t', 1)
                        # Clean up class name - take first name if there are multiple
                        clean_name = class_name.split(',')[0].strip()
                        class_mappings[class_id] = clean_name
        
        print(f"Loaded {len(class_mappings)} class mappings from words.txt")
        return class_mappings
    
    def _get_available_classes(self) -> List[str]:
        """Get list of available class directories"""
        split_dir = self.root_dir / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} not found")
        
        # Get all class directories (starting with 'n')
        class_dirs = [d.name for d in split_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('n')]
        
        # Sort for consistency
        class_dirs.sort()
        
        return class_dirs
    
    def _load_samples(self):
        """Load image samples for available classes"""
        split_dir = self.root_dir / self.split
        
        for class_id in self.available_classes:
            class_dir = split_dir / class_id
            if not class_dir.exists():
                print(f"Warning: Class directory {class_dir} not found")
                continue
            
            # Load all JPEG images
            image_files = list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.jpg'))
            
            for img_path in image_files:
                self.samples.append((str(img_path), class_id))
        
        print(f"Loaded {len(self.samples)} samples from {len(self.available_classes)} classes")
    
    def _compute_text_embeddings(self) -> Dict[str, torch.Tensor]:
        """Pre-compute CLIP text embeddings for all classes"""
        embeddings = {}
        device = next(self.clip_model.parameters()).device
        
        print("Computing CLIP text embeddings...")
        
        with torch.no_grad():
            for class_id in tqdm(self.available_classes, desc="Computing embeddings"):
                class_name = self.class_to_name.get(class_id, f"class {class_id}")
                
                # Create descriptive text prompt
                text_prompt = f"a photo of a {class_name}"
                
                # Tokenize and encode text
                text_tokens = clip.tokenize([text_prompt]).to(device)
                text_embedding = self.clip_model.encode_text(text_tokens)
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                
                embeddings[class_id] = text_embedding.cpu().squeeze(0)
        
        return embeddings
    
    def get_class_info(self, class_id: str) -> Dict[str, str]:
        """Get human-readable information about a class"""
        class_name = self.class_to_name.get(class_id, "unknown")
        return {
            'class_id': class_id,
            'class_name': class_name,
            'text_prompt': f"a photo of a {class_name}"
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # Get text embedding
            text_embedding = self.text_embeddings[class_id]
            class_name = self.class_to_name.get(class_id, f"class {class_id}")
            
            return image, text_embedding, class_name
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random sample instead
            return self.__getitem__((idx + 1) % len(self.samples))

def create_imagenet_mini_loaders(root_dir: str, clip_model, batch_size: int = 4, 
                                image_size: int = 256, num_workers: int = 4,
                                max_classes: Optional[int] = None):
    """Create training and validation data loaders for ImageNet Mini"""
    
    # Training transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    print(f"Creating ImageNet Mini datasets from {root_dir}")
    
    train_dataset = ImageNetMiniDataset(
        root_dir, clip_model, train_transform, 'train', max_classes
    )
    
    # Check if val split exists, otherwise use train split for validation
    val_split = 'val' if (Path(root_dir) / 'val').exists() else 'train'
    if val_split == 'train':
        print("No separate validation split found, using training data for validation")
    
    val_dataset = ImageNetMiniDataset(
        root_dir, clip_model, val_transform, val_split, max_classes
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Created data loaders:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    
    return train_loader, val_loader, train_dataset.available_classes

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def compute_loss(model: nn.Module, x_0: torch.Tensor, text_embeddings: torch.Tensor, 
                scheduler: DDIMScheduler, device: torch.device) -> torch.Tensor:
    """Compute diffusion training loss"""
    batch_size = x_0.size(0)
    
    # Sample random timesteps
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()
    
    # Sample noise
    noise = torch.randn_like(x_0)
    
    # Add noise to images
    noisy_images = scheduler.add_noise(x_0, noise, timesteps)
    
    # Predict noise
    predicted_noise = model(noisy_images, timesteps, text_embeddings)
    
    # Compute loss
    loss = F.mse_loss(predicted_noise, noise)
    
    return loss

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                scheduler: DDIMScheduler, num_epochs: int, device: torch.device,
                save_dir: Path, learning_rate: float = 1e-4):
    """Training loop"""
    
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(save_dir / "logs")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_loader), eta_min=1e-6
    )
    
    global_step = 0
    best_val_loss = float('inf')
    
    print(f"Starting training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for batch_idx, (images, text_embeddings, class_names) in enumerate(pbar):
            images = images.to(device)
            text_embeddings = text_embeddings.to(device)
            
            # Compute loss
            loss = compute_loss(model, images, text_embeddings, scheduler, device)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            
            epoch_losses.append(loss.item())
            
            if global_step % 100 == 0:
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
            
            global_step += 1
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for images, text_embeddings, class_names in val_loader:
                images = images.to(device)
                text_embeddings = text_embeddings.to(device)
                
                val_loss = compute_loss(model, images, text_embeddings, scheduler, device)
                val_losses.append(val_loss.item())
        
        avg_train_loss = np.mean(epoch_losses)
        avg_val_loss = np.mean(val_losses)
        
        writer.add_scalar('Epoch/TrainLoss', avg_train_loss, epoch)
        writer.add_scalar('Epoch/ValLoss', avg_val_loss, epoch)
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = save_dir / "best_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'model_config': {
                    'text_embed_dim': 512,
                    'model_channels': 128,
                    'image_size': 256,
                    'in_channels': 3
                }
            }, best_model_path)
    
    # Save final model
    final_model_path = save_dir / "ddim_imagenet.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'val_loss': best_val_loss,
        'model_config': {
            'text_embed_dim': 512,
            'model_channels': 128,
            'image_size': 256,
            'in_channels': 3
        }
    }, final_model_path)
    
    writer.close()
    return str(final_model_path)

# =============================================================================
# DDIM SAMPLING AND DEMO
# =============================================================================

@torch.no_grad()
def ddim_sample(model: nn.Module, scheduler: DDIMScheduler, text_embeddings: torch.Tensor,
                num_inference_steps: int = 20, eta: float = 0.0, cfg_scale: float = 7.5,
                device: torch.device = None) -> torch.Tensor:
    """Generate images using DDIM sampling"""
    
    batch_size = text_embeddings.size(0)
    image_shape = (batch_size, 3, 256, 256)
    
    # For classifier-free guidance
    if cfg_scale > 1.0:
        # Create unconditional embedding (zeros)
        uncond_embeddings = torch.zeros_like(text_embeddings)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)
        
    # Initialize with random noise
    if cfg_scale > 1.0:
        images = torch.randn((batch_size * 2, 3, 256, 256), device=device)
    else:
        images = torch.randn(image_shape, device=device)
    
    # Set timesteps
    scheduler.set_timesteps(num_inference_steps, device)
    
    model.eval()
    
    for i, t in enumerate(tqdm(scheduler.timesteps, desc="DDIM Sampling")):
        # Expand timestep to batch dimension
        timestep_batch = t.expand(images.shape[0])
        
        # Predict noise
        noise_pred = model(images, timestep_batch, text_embeddings.repeat(images.shape[0] // text_embeddings.shape[0], 1))
        
        # Apply classifier-free guidance
        if cfg_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            images = images[:batch_size]  # Keep only conditional batch
        
        # DDIM step
        images = scheduler.step(noise_pred, t.item(), images, eta=eta)
        
        # Update images for next iteration if using CFG
        if cfg_scale > 1.0:
            images = torch.cat([images, images], dim=0)
    
    # Return only the conditional batch
    if cfg_scale > 1.0:
        images = images[:batch_size]
    
    return images

class DDIMDemo:
    """Interactive demo for DDIM text-to-image generation"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        
        # Load CLIP model
        print("Loading CLIP model...")
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        
        # Load diffusion model
        print(f"Loading diffusion model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model_config = checkpoint.get('model_config', {
            'text_embed_dim': 512,
            'model_channels': 128,
            'image_size': 256,
            'in_channels': 3
        })
        
        self.model = TextConditionalUNet(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize DDIM scheduler
        self.scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
        
        print("DDIM Demo initialized successfully!")
    
    def encode_text(self, text_prompt: str) -> torch.Tensor:
        """Encode text prompt using CLIP"""
        with torch.no_grad():
            text_tokens = clip.tokenize([text_prompt]).to(self.device)
            text_embedding = self.clip_model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding
    
    def generate_image(self, text_prompt: str, num_inference_steps: int = 20,
                      eta: float = 0.0, cfg_scale: float = 7.5, 
                      seed: Optional[int] = None) -> torch.Tensor:
        """Generate image from text prompt using DDIM"""
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Encode text
        text_embedding = self.encode_text(text_prompt)
        
        # Generate image
        images = ddim_sample(
            self.model, self.scheduler, text_embedding,
            num_inference_steps=num_inference_steps,
            eta=eta, cfg_scale=cfg_scale, device=self.device
        )
        
        return images[0]  # Return first image
    
    def denormalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Convert from [-1, 1] to [0, 1] range"""
        return torch.clamp((image + 1) / 2, 0, 1)
    
    def tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image"""
        image_tensor = self.denormalize_image(image_tensor)
        image_np = image_tensor.cpu().permute(1, 2, 0).numpy()
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        return image_pil
    
    def compute_clip_similarity(self, generated_image: torch.Tensor, text_prompt: str) -> float:
        """Compute CLIP similarity between generated image and text prompt"""
        
        # Prepare image for CLIP
        image = self.denormalize_image(generated_image)
        
        clip_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        image_input = clip_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get image and text features
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.encode_text(text_prompt)
            
            # Normalize and compute similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = torch.cosine_similarity(image_features, text_features).item()
        
        return similarity

def generate_image_demo(
    model_path: str = "checkpoints/ddim_imagenet.pth",
    text_prompt: str = "a photo of a golden retriever",
    output_dir: str = "demo_outputs/",
    device: str = "auto",
    cfg_scale: float = 7.5,
    num_inference_steps: int = 20,
    eta: float = 0.0,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    DDIM demo function for fast text-to-image generation
    
    Args:
        model_path: Path to trained model checkpoint
        text_prompt: Text description for image generation
        output_dir: Directory to save outputs
        device: Device to use ('auto', 'cuda', 'cpu')
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
        num_inference_steps: Number of DDIM denoising steps (10-50)
        eta: DDIM sampling noise factor (0.0 = deterministic, 1.0 = stochastic)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with generated image and metadata
    """
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize demo
    demo = DDIMDemo(model_path, device)
    
    # Generate image
    print(f"\nGenerating image for prompt: '{text_prompt}'")
    print(f"CFG Scale: {cfg_scale}, Steps: {num_inference_steps}, eta: {eta}")
    if seed is not None:
        print(f"Seed: {seed}")
    
    start_time = time.time()
    generated_image = demo.generate_image(
        text_prompt=text_prompt,
        num_inference_steps=num_inference_steps,
        eta=eta,
        cfg_scale=cfg_scale,
        seed=seed
    )
    generation_time = time.time() - start_time
    
    # Compute CLIP similarity
    clip_similarity = demo.compute_clip_similarity(generated_image, text_prompt)
    
    # Create safe filename
    safe_prompt = "".join(c for c in text_prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt.replace(' ', '_')[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_prompt}_{timestamp}.png"
    
    # Save image
    final_pil = demo.tensor_to_pil(generated_image)
    image_path = output_path / filename
    final_pil.save(image_path)
    
    # Save metadata
    metadata = {
        'text_prompt': text_prompt,
        'clip_similarity': clip_similarity,
        'cfg_scale': cfg_scale,
        'num_inference_steps': num_inference_steps,
        'eta': eta,
        'seed': seed,
        'generation_time': generation_time,
        'model_path': model_path,
        'timestamp': timestamp
    }
    
    metadata_path = output_path / f"{safe_prompt}_{timestamp}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGeneration completed in {generation_time:.2f} seconds")
    print(f"CLIP Similarity Score: {clip_similarity:.3f}")
    print(f"Image saved to: {image_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    return {
        'generated_image': generated_image,
        'clip_similarity': clip_similarity,
        'generation_time': generation_time,
        'metadata': metadata,
        'image_path': str(image_path),
        'metadata_path': str(metadata_path)
    }

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def main_train():
    """Main training function"""
    
    # Configuration
    config = {
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'image_size': 256,
        'save_dir': Path('./checkpoints'),
        'imagenet_root': r'D:\DDPM-diffusion\data\imagenet-mini',  # Local ImageNet Mini dataset
        'max_classes': 100,  # Limit to first 100 classes for faster training (optional)
        'num_workers': 2,
        'gradient_clip': 1.0,
        'save_every': 5,
        'demo_every': 10,
        'guidance_scale': 7.5,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    # Create data loaders for ImageNet Mini
    train_loader, val_loader, available_classes = create_imagenet_mini_loaders(
        root_dir=config['imagenet_root'],
        clip_model=clip_model,
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        num_workers=config['num_workers'],
        max_classes=config.get('max_classes', None)
    )
    
    # Create model
    model = TextConditionalUNet(
        text_embed_dim=512,
        model_channels=128,
        image_size=config['image_size'],
        in_channels=3
    ).to(device)
    
    # Create DDIM scheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    
    # Train the model
    final_model_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        num_epochs=config['num_epochs'],
        device=device,
        save_dir=config['save_dir'],
        learning_rate=config['learning_rate']
    )
    
    print(f"Training completed! Model saved to: {final_model_path}")

def main_demo():
    """Main demo function"""
    
    parser = argparse.ArgumentParser(description="DDIM Text-to-Image Demo")
    parser.add_argument("--model_path", type=str, default="checkpoints/ddim_imagenet.pth",
                       help="Path to trained model checkpoint")
    parser.add_argument("--prompt", type=str, default="a photo of a golden retriever",
                       help="Text prompt for image generation")
    parser.add_argument("--output_dir", type=str, default="demo_outputs/",
                       help="Output directory for generated images")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use for generation")
    parser.add_argument("--cfg_scale", type=float, default=7.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--steps", type=int, default=20,
                       help="Number of DDIM denoising steps")
    parser.add_argument("--eta", type=float, default=0.0,
                       help="DDIM sampling noise factor (0.0=deterministic)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--batch", nargs="+", type=str,
                       help="Generate multiple prompts in batch")
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch generation
        for prompt in args.batch:
            generate_image_demo(
                model_path=args.model_path,
                text_prompt=prompt,
                output_dir=args.output_dir,
                device=args.device,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.steps,
                eta=args.eta,
                seed=args.seed
            )
    else:
        # Single generation
        generate_image_demo(
            model_path=args.model_path,
            text_prompt=args.prompt,
            output_dir=args.output_dir,
            device=args.device,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.steps,
            eta=args.eta,
            seed=args.seed
        )

def test_dataset():
    """Test ImageNet Mini dataset loading"""
    print("Testing ImageNet Mini dataset loading...")
    
    # Load CLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    # Test data loader creation
    dataset_path = r'D:\DDPM-diffusion\data\imagenet-mini'
    
    try:
        train_loader, val_loader, available_classes = create_imagenet_mini_loaders(
            root_dir=dataset_path,
            clip_model=clip_model,
            batch_size=2,
            image_size=256,
            num_workers=0,  # Use 0 for testing
            max_classes=10  # Test with just 10 classes
        )
        
        print(f"\n✅ Dataset loading successful!")
        print(f"Available classes (first 10): {available_classes[:10]}")
        
        # Test loading a batch
        print("\nTesting batch loading...")
        for batch_idx, (images, text_embeddings, class_names) in enumerate(train_loader):
            print(f"  Batch {batch_idx + 1}:")
            print(f"    Images shape: {images.shape}")
            print(f"    Text embeddings shape: {text_embeddings.shape}")
            print(f"    Class names: {class_names}")
            
            if batch_idx >= 2:  # Test just a few batches
                break
        
        print("\n✅ Dataset test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            main_train()
        elif sys.argv[1] == "test":
            test_dataset()
        else:
            main_demo()
    else:
        main_demo()