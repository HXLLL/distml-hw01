import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Shape: [B, E, H/P, W/P]
        x = x.flatten(2)  # Shape: [B, E, N] where N is number of patches
        x = x.transpose(1, 2)  # Shape: [B, N, E]
        return x

class ViT(nn.Module):
    def __init__(self, num_classes=100, emb_size=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size=emb_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.positional_embedding = nn.Parameter(torch.zeros(1, (32 // 16) ** 2 + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, dim_feedforward=int(emb_size * mlp_ratio), dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embedding
        x = self.dropout(x)

        x = self.transformer_encoder(x)
        x = x[:, 0]
        x = self.mlp_head(x)
        return x

def vit():
    return ViT()
