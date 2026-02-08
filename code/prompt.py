import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MemoryPool(nn.Module):
    def __init__(self, num_keys, key_dim, value_dim):
        super().__init__()
        self.keys = nn.Parameter(torch.empty(num_keys, key_dim))
        self.values = nn.Parameter(torch.empty(num_keys, value_dim))
        nn.init.xavier_uniform_(self.keys)
        nn.init.xavier_uniform_(self.values)

    def forward(self, query):
        """ query: (..., K) """
        logits = torch.einsum('...d,kd->...k', query, self.keys)
        weights = F.softmax(logits, dim=-1)
        return torch.einsum('...k,kv->...v', weights, self.values)

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_keys, key_dim, value_dim):
        super().__init__()
        self.temporal_embedding = nn.Embedding(30, output_dim)
        self.attention = nn.Linear(input_dim, output_dim)
        self.scale = (output_dim ** -0.5)
        self.memory_pool = MemoryPool(num_keys, key_dim, value_dim)

    def forward(self, x):
        """ x: (B, T, H, W, C) """
        B, T, H, W, C = x.shape
        
  
        time_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)  # [B, T]
        time_emb = self.temporal_embedding(time_ids)  # [B, T, D]
        
     
        x_flat = rearrange(x, 'b t h w c -> (b t) (h w) c')  # [B*T, H*W, C]
        time_emb = rearrange(time_emb, 'b t d -> (b t) () d')  # [B*T, 1, D]
        

        attn = self.attention(x_flat + time_emb) * self.scale
        attn = F.softmax(attn, dim=1)
        

        prompt = rearrange(attn * x_flat, '(b t) n d -> b t n d', b=B)
        return self.memory_pool(prompt).reshape(B, T, H, W,-1)

class SpatialAttention(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, num_keys, key_dim, value_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size, 
                     padding=kernel_size//2, groups=8),
            nn.BatchNorm2d(output_dim),
            nn.GELU()
        )
        self.memory_pool = MemoryPool(num_keys, key_dim, value_dim)

    def forward(self, x):
        """ x: (B, T, H, W, C) """
        B, T, H, W, C = x.shape
        

        x = rearrange(x, 'b t h w c -> (b t) c h w')
        spatial_attn = self.conv(x)  # [B*T, C, H, W]
        

        spatial_attn = rearrange(spatial_attn, 'b c h w -> b (h w) c')
        
        return self.memory_pool(spatial_attn).reshape(B, T, H, W, -1)

class SpatioTemporalPromptGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, key_dim, num_keys, value_dim):
        super().__init__()
        self.spatial_att = SpatialAttention(input_dim, output_dim, kernel_size, 
                                          num_keys, key_dim, value_dim)
        self.temporal_att = TemporalAttention(input_dim, output_dim,
                                             num_keys, key_dim, value_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2*value_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, x):
        """ x: (B, T, H, W, C) """
        ps = self.spatial_att(x)
        pt = self.temporal_att(x)

        return self.fusion(torch.cat([ps, pt], dim=-1))
