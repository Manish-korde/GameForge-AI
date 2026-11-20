import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class Block(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.silu = nn.SiLU()
    def forward(self, x, t):
        h = self.silu(self.norm1(self.conv1(x)))
        time_emb = self.silu(self.time_mlp(t))
        h = h + time_emb[(..., ) + (None, ) * 2]
        h = self.silu(self.norm2(self.conv2(h)))
        return h

class ActionConditionedUNet(nn.Module):
    """
    Rung 2 Minimal Action-Conditioned U-Net for Experiment C.
    Extends ReferenceConditionedUNet by exactly 256 parameters:
    self.action_embed = nn.Embedding(2, 128)
    combined_emb = time_emb + self.action_embed(action_id)
    """
    def __init__(self, time_dim=128, num_actions=2):
        super().__init__()
        # Timestep MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # Minimal action conditioning (+256 parameters)
        self.action_embed = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=time_dim
        )
        
        # Input: 4 ch noisy target (x_t) + 4 ch clean reference (x_ref) = 8 channels
        self.in_conv = nn.Conv2d(8, 32, 3, padding=1)
        
        # Down stages
        self.b1 = Block(32, 64, time_dim)
        self.down1 = nn.MaxPool2d(2)
        
        self.b2 = Block(64, 128, time_dim)
        self.down2 = nn.MaxPool2d(2)
        
        # Mid stage
        self.mid = Block(128, 256, time_dim)
        
        # Up stages
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.b_up2 = Block(256, 128, time_dim)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.b_up1 = Block(128, 64, time_dim)
        
        # Output: 4 channels (predicting epsilon or velocity v)
        self.out_conv = nn.Conv2d(64, 4, 3, padding=1)

    def forward(self, x_t, t, x_ref, action_id):
        """
        Forward pass with joint reference and discrete action conditioning.
        x_t: [B, 4, 64, 64]
        t: [B]
        x_ref: [B, 4, 64, 64]
        action_id: [B] long tensor in {0, 1}
        """
        # 1. 8-channel early spatial concatenation
        x = torch.cat([x_t, x_ref], dim=1)
        
        # 2. Combined temporal + action conditioning
        time_emb = self.time_mlp(t)
        act_emb = self.action_embed(action_id)
        combined_emb = time_emb + act_emb
        
        # 3. U-Net backbone
        h0 = self.in_conv(x)
        h1 = self.b1(h0, combined_emb)
        d1 = self.down1(h1)
        
        h2 = self.b2(d1, combined_emb)
        d2 = self.down2(h2)
        
        m = self.mid(d2, combined_emb)
        
        u2 = self.up2(m)
        cat2 = torch.cat([u2, h2], dim=1)
        dec2 = self.b_up2(cat2, combined_emb)
        
        u1 = self.up1(dec2)
        cat1 = torch.cat([u1, h1], dim=1)
        dec1 = self.b_up1(cat1, combined_emb)
        
        return self.out_conv(dec1)

if __name__ == '__main__':
    model = ActionConditionedUNet()
    total_params = sum(p.numel() for p in model.parameters())
    action_params = sum(p.numel() for p in model.action_embed.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Action embedding parameters: {action_params:,}")
    
    # Test forward
    xt = torch.randn(2, 4, 64, 64)
    xref = torch.randn(2, 4, 64, 64)
    t = torch.tensor([100, 500])
    a = torch.tensor([0, 1])
    out = model(xt, t, xref, a)
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 4, 64, 64)
    print("ActionConditionedUNet validation PASSED!")

