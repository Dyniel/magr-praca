#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from skimage.segmentation import slic, relabel_sequential
from skimage.util import img_as_float
from PIL import Image
from tqdm import tqdm
from scipy import linalg
import wandb

# ==================== KONFIGURACJA ====================
config = {
    "image_size": 256,
    "batch_size": 16,
    "z_dim": 256,
    "num_superpixels": 150,
    "g_channels": 128,
    "d_channels": 64,
    "g_lr": 5e-5,  # !!! Zmniejszone LR
    "d_lr": 2e-4,
    "beta1": 0.0,
    "beta2": 0.99,
    "r1_gamma": 5.0,  # !!! Mniejsza kara R1
    "r2_gamma": 0.1,
    "num_epochs": 200,
    "dataset_path": "/home/student2/histo/data/lung_colon_image_set/lung_image_sets/lung_scc/train",
    "cache_dir": "/home/student2/histo/superpixel_cache",
    "num_workers": 8,
    "mixed_precision": True,
    "ada_in": False,
    "d_steps_per_g": 2,  # !!! Więcej kroków D
    "dropout_rate": 0.2,  # !!! Dodany dropout
    "debug_num_images": 0
}

wandb.init(project="MedicalR3GAN", config=config)
cfg = wandb.config


# ==================== UTILS ====================
def spectral_norm(layer):
    return nn.utils.spectral_norm(layer)  # !!! Spectral norm


def relabel_and_clip(seg, max_labels):
    seg, _, _ = relabel_sequential(seg)
    seg = np.where(seg < max_labels, seg, max_labels - 1)
    return seg.astype(np.int32)


def create_adjacency_matrix(seg):
    H, W = seg.shape
    S = cfg.num_superpixels
    adj = np.zeros((S, S), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            c = int(seg[y, x])
            if c >= S: continue
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < H and 0 <= nx < W:
                    n = int(seg[ny, nx])
                    if n < S and n != c:
                        adj[c, n] = adj[n, c] = 1.0
    deg = adj.sum(1)
    inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    inv_sqrt[mask] = deg[mask] ** -0.5
    D = np.diag(inv_sqrt)
    return D @ adj @ D


def precompute_superpixels(paths):
    if os.path.exists(cfg.cache_dir):
        shutil.rmtree(cfg.cache_dir)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    for p in tqdm(paths, desc="Preprocessing"):
        name = os.path.splitext(os.path.basename(p))[0]
        fn = os.path.join(cfg.cache_dir, name + ".npz")
        if os.path.exists(fn): continue
        img = Image.open(p).convert("RGB")
        img = img.resize((cfg.image_size, cfg.image_size), Image.BILINEAR)
        seg = slic(img_as_float(img), n_segments=cfg.num_superpixels, compactness=10)
        seg = relabel_and_clip(seg, cfg.num_superpixels)
        adj = create_adjacency_matrix(seg)
        np.savez(fn, segments=seg, adj=adj)


# ==================== MODEL ====================
class WSConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel, stride, padding))  # !!! Spectral
        self.scale = math.sqrt(2 / (in_ch * kernel * kernel))
        nn.init.normal_(self.conv.weight, 0, 1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x * self.scale)


class AdaIN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            WSConv2d(dim, dim * 2, 1, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(dim * 2, dim * 2, 1, 1, 0)
        )

    def forward(self, x):
        style = self.mlp(x.mean([2, 3], keepdim=True))
        gamma, beta = style.chunk(2, 1)
        return gamma * x + beta


class GCNBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = WSConv2d(dim, dim, 1, 1, 0)
        self.ada = AdaIN(dim) if cfg.ada_in else None
        self.dropout = nn.Dropout(cfg.dropout_rate)  # !!! Dropout

    def forward(self, x, A):
        B, C, S, _ = x.shape
        x = self.proj(x)
        feat = x.squeeze(-1).permute(0, 2, 1)
        feat = torch.einsum('bij,bjk->bik', A, feat)
        x = feat.permute(0, 2, 1).unsqueeze(-1)
        if self.ada: x = self.ada(x)
        return self.dropout(F.leaky_relu(x, 0.2))  # !!! Dropout


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Projekcja wejścia (obraz + szum)
        self.proj = WSConv2d(3 + cfg.z_dim, cfg.g_channels, 1, 1, 0)
        # 8 bloków GCN
        self.blocks = nn.ModuleList([GCNBlock(cfg.g_channels) for _ in range(8)])
        # Lekki decoder – bez zmiany rozmiaru przestrzennego
        self.upsample = nn.Sequential(
            nn.Conv2d(cfg.g_channels, cfg.g_channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.g_channels // 2, 3, 3, padding=1)
        )
        # Normalizacja wyjścia
        self.final_norm = nn.InstanceNorm2d(3)

    def forward(self, z, images, segments, A):
        """
        z:        [B, z_dim]
        images:   [B, 3, H, W]
        segments: [B, H, W]   (int od 0 do S-1)
        A:        [B, S, S]
        """
        B, _, H, W = images.shape
        N = H * W
        S = cfg.num_superpixels

        # 1) Agregacja cech piksel→superpiksel
        flat     = images.view(B, 3, N)                         # [B, 3, N]
        seg_flat = segments.view(B, N)                          # [B, N]
        one_hot  = F.one_hot(seg_flat, S).float()               # [B, N, S]
        counts   = one_hot.sum(dim=1).unsqueeze(-1)              # [B, S, 1]
        feats    = (one_hot.transpose(1,2) @ flat.transpose(1,2)) / (counts + 1e-6)
        #         → [B, S, 3]

        # 2) Doklejenie z do superpikseli
        z_rep = z.view(B, 1, cfg.z_dim).expand(-1, S, -1)       # [B, S, z_dim]
        x     = torch.cat([feats, z_rep], dim=2)                # [B, S, 3+z_dim]
        x     = x.permute(0,2,1).unsqueeze(-1)                  # [B, 3+z, S, 1]

        # 3) Przepuszczenie przez GCN-y
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x, A)                                       # [B, g_channels, S, 1]

        # 4) Rozrzucenie cech superpiksel→piksel
        gC        = x.size(1)                                  # = cfg.g_channels
        sup_feats = x.squeeze(-1).permute(0,2,1)               # [B, S, gC]
        pix_feats = one_hot @ sup_feats                        # [B, N, gC]
        pix_feats = pix_feats.permute(0,2,1).view(B, gC, H, W)  # [B, gC, H, W]

        # 5) Lekki decoder – bez upsamplingu przestrzennego
        out = self.upsample(pix_feats)                          # [B, 3, 256, 256]
        out = self.final_norm(out)

        return torch.tanh(out)  # [B, 3, 256, 256]


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        d = cfg.d_channels
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(3, d, 3, padding=1)),  # !!! Spectral
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(d, d * 2, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(d * 2, d * 4, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            spectral_norm(nn.Linear(d * 4, 1))  # !!! Spectral
        )

    def forward(self, x):
        return self.main(x).squeeze(1)


# ==================== DATASET ====================
class MedicalDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        precompute_superpixels(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        name = os.path.splitext(os.path.basename(p))[0]
        data = np.load(os.path.join(cfg.cache_dir, name + ".npz"))
        seg = torch.from_numpy(data["segments"]).long()
        adj = torch.from_numpy(data["adj"]).float()
        img = Image.open(p).convert("RGB")
        img = img.resize((cfg.image_size, cfg.image_size), Image.BILINEAR)
        img = self.tf(img)
        return {"image": img, "segments": seg, "adj": adj}


# ==================== TRENER ====================
class Trainer:
    def __init__(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(cfg.cache_dir):
            shutil.rmtree(cfg.cache_dir)
        os.makedirs(cfg.cache_dir, exist_ok=True)
        self.G = Generator().to(self.dev)
        self.D = Discriminator().to(self.dev)
        self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg.g_lr, betas=(cfg.beta1, cfg.beta2))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg.d_lr, betas=(cfg.beta1, cfg.beta2))
        paths = [os.path.join(cfg.dataset_path, f) for f in os.listdir(cfg.dataset_path)]
        if cfg.debug_num_images > 0:
            paths = paths[:cfg.debug_num_images]
        ds = MedicalDataset(paths)
        self.loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                 num_workers=cfg.num_workers, pin_memory=True)
        self.incep = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                                  aux_logits=True).eval().to(self.dev)

    def rel_loss(self, r, f):
        return (F.softplus(-r + f) + F.softplus(f - r)).mean()

    def grad_pen(self, real):
        real.requires_grad_(True)
        d_out = self.D(real)
        grads = torch.autograd.grad(d_out.sum(), real, create_graph=True)[0]
        return ((grads.flatten(1).norm(2, 1) - 1) ** 2).mean()

    def calculate_fid(self, real, fake):
        with torch.no_grad():
            fr = self.incep(real).cpu().numpy()
            ff = self.incep(fake).cpu().numpy()
        mu1, s1 = fr.mean(0), np.cov(fr, rowvar=False)
        mu2, s2 = ff.mean(0), np.cov(ff, rowvar=False)
        diff = mu1 - mu2
        covmean = linalg.sqrtm(s1.dot(s2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return diff.dot(diff) + np.trace(s1 + s2 - 2 * covmean)

    def train_epoch(self, epoch):
        self.G.train();
        self.D.train()
        pbar = tqdm(self.loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            real = batch["image"].to(self.dev)
            seg = batch["segments"].to(self.dev)
            adj = batch["adj"].to(self.dev)
            B = real.size(0)
            z = torch.randn(B, cfg.z_dim, device=self.dev)

            # !!! Więcej kroków dla D
            for _ in range(cfg.d_steps_per_g):
                self.optD.zero_grad()
                fake = self.G(z, real, seg, adj).detach()
                r_scr = self.D(real)
                f_scr = self.D(fake)
                lossD = self.rel_loss(r_scr, f_scr) + cfg.r1_gamma * self.grad_pen(real)
                lossD.backward();
                self.optD.step()

            # Generator step
            self.optG.zero_grad()
            fake = self.G(z, real, seg, adj)
            lossG = -self.D(fake).mean()
            lossG.backward();
            self.optG.step()

            pbar.set_postfix(D_loss=lossD.item(), G_loss=lossG.item())
            wandb.log({"D_loss": lossD.item(), "G_loss": lossG.item()})

    def evaluate(self, epoch):
        self.G.eval()
        with torch.no_grad():
            # pobieramy pierwszy batch
            batch = next(iter(self.loader))
            real = batch["image"].to(self.dev)      # [B,3,H,W]
            seg  = batch["segments"].to(self.dev)   # [B,H,W]
            adj  = batch["adj"].to(self.dev)       # [B,S,S]

            B = real.size(0)                        # faktyczny rozmiar batcha (np. 5)
            k = min(10, B)                          # ile próbek wyświetlić

            real = real[:k]
            seg  = seg[:k]
            adj  = adj[:k]

            # teraz z ma k wierszy, nie więcej niż B
            z = torch.randn(k, cfg.z_dim, device=self.dev)

            fake = self.G(z, real, seg, adj)       # [k,3,256,256]
            grid = torchvision.utils.make_grid(fake, nrow=k, normalize=True)
            wandb.log({"samples": [wandb.Image(grid)]})

    def train(self):
        for ep in range(cfg.num_epochs):
            self.train_epoch(ep)
            self.evaluate(ep)
            if ep % 10 == 0:
                torch.save(self.G.state_dict(), f"checkpoint_G_{ep}.pt")
                torch.save(self.D.state_dict(), f"checkpoint_D_{ep}.pt")


if __name__ == "__main__":
    Trainer().train()