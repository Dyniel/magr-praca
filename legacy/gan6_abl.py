#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DCGAN baseline (ablation study) bez zależności od torchvision oraz z opcjonalnym Weights & Biases.
"""
import math, random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Próba importu wandb, jeśli nie działa, wyłącz logowanie
try:
    import wandb
    USE_WANDB = True
except Exception:
    print("Warning: wandb not available, logging disabled.")
    USE_WANDB = False

# ==================== CONFIG ====================
class Cfg:
    dataset_path   = "/home/student2/histo/data/lung_colon_image_set/lung_image_sets/lung_scc/train"
    image_size     = 256
    batch_size     = 8
    num_workers    = 4
    z_dim          = 128
    gen_feat       = 512
    d_feat         = 64
    lr             = 2e-4
    beta1, beta2   = 0.0, 0.99
    r1_gamma       = 10.0
    num_epochs     = 200
    log_every      = 100

cfg = Cfg()

# ==================== WEIGHTS & BIASES ====================
if USE_WANDB:
    try:
        wandb.init(
            project="graph_gan_ablation",
            config={
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "beta1": cfg.beta1,
                "beta2": cfg.beta2,
                "z_dim": cfg.z_dim,
                "gen_feat": cfg.gen_feat,
                "d_feat": cfg.d_feat,
            }
        )
        wandb.config.update({"image_size": cfg.image_size, "epochs": cfg.num_epochs})
    except Exception as e:
        print(f"Warning: wandb.init failed: {e}")
        USE_WANDB = False

# ==================== DATASET (bez torchvision) ====================
class ImageDataset(Dataset):
    def __init__(self, root):
        root = Path(root)
        exts = {".png", ".jpg", ".jpeg"}
        self.paths = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        pil = Image.open(p).convert("RGB")
        # Ablation: ręczne resize + normalizacja
        pil = pil.resize((cfg.image_size, cfg.image_size), Image.BILINEAR)
        arr = np.asarray(pil).astype(np.float32) / 127.5 - 1.0  # [-1,1]
        # zamiana do Tensor: H×W×C -> C×H×W
        img = torch.from_numpy(arr).permute(2, 0, 1)
        return img

def collate(batch):
    return torch.stack(batch)

# ==================== GENERATOR ====================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_size = 4
        n_up = int(math.log2(cfg.image_size / self.init_size))
        self.proj = nn.Linear(cfg.z_dim, cfg.gen_feat * self.init_size**2)
        feats = [cfg.gen_feat // (2**i) for i in range(n_up+1)]
        self.ups = nn.ModuleList()
        for i in range(n_up):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(feats[i], feats[i+1], 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(feats[i+1], feats[i+1], 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            ))
        self.to_rgb = nn.Conv2d(feats[-1], 3, 1)

    def forward(self, batch_size):
        z_noise = torch.randn(batch_size, cfg.z_dim, device=next(self.parameters()).device)
        x = self.proj(z_noise).view(batch_size, cfg.gen_feat, self.init_size, self.init_size)
        for up in self.ups:
            x = up(x)
        return torch.tanh(self.to_rgb(x))

# ==================== DISCRIMINATOR ====================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_ch = 3
        feats = [cfg.d_feat, cfg.d_feat*2, cfg.d_feat*4, cfg.d_feat*8]
        for f in feats:
            layers += [nn.Conv2d(in_ch, f, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            in_ch = f
        self.net = nn.Sequential(*layers)
        n_down = len(feats)
        final_sz = cfg.image_size // (2**n_down)
        fc_in = feats[-1] * final_sz * final_sz
        self.fc = nn.Linear(fc_in, 1)

    def forward(self, img):
        x = self.net(img)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(1)

# ==================== TRAINER ====================
class Trainer:
    def __init__(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = Generator().to(self.dev)
        self.D = Discriminator().to(self.dev)
        self.optG = torch.optim.Adam(self.G.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

        ds = ImageDataset(cfg.dataset_path)
        self.loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                 num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True)
        self.bce = nn.BCEWithLogitsLoss()

    def r1_penalty(self, real_img, real_pred):
        (grad,) = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
        return grad.flatten(1).pow(2).sum(1).mean()

    def train(self):
        step = 0
        for ep in range(cfg.num_epochs):
            pbar = tqdm(self.loader, desc=f"Epoch {ep}")
            for imgs in pbar:
                B = imgs.size(0)
                imgs = imgs.to(self.dev)

                # Discriminator
                self.optD.zero_grad(set_to_none=True)
                imgs.requires_grad_()
                r_log = self.D(imgs)
                r_loss = self.bce(r_log, torch.ones_like(r_log))
                with torch.no_grad():
                    f_imgs = self.G(B)
                f_log = self.D(f_imgs)
                f_loss = self.bce(f_log, torch.zeros_like(f_log))
                r1 = self.r1_penalty(imgs, r_log)
                d_loss = 0.5 * (r_loss + f_loss) + 0.5 * cfg.r1_gamma * r1
                d_loss.backward()
                self.optD.step()

                # Generator
                self.optG.zero_grad(set_to_none=True)
                f_imgs = self.G(B)
                g_log = self.D(f_imgs)
                g_loss = self.bce(g_log, torch.ones_like(g_log))
                g_loss.backward()
                self.optG.step()

                # Log do W&B jeśli dostępne
                if USE_WANDB and step % cfg.log_every == 0:
                    imgs_for_log = []
                    for img in f_imgs.detach().cpu():
                        arr = ((img.permute(1,2,0).numpy() + 1) * 127.5).clip(0,255).astype(np.uint8)
                        imgs_for_log.append(Image.fromarray(arr))
                    wandb.log({
                        "d_loss": d_loss.item(),
                        "g_loss": g_loss.item(),
                        "r1_penalty": r1.item(),
                        "generated_images": [wandb.Image(im) for im in imgs_for_log],
                        "epoch": ep,
                        "step": step
                    }, step=step)
                step += 1

            if ep % 10 == 0:
                torch.save(self.G.state_dict(), f"gen_ablation_ep{ep}.pt")
                torch.save(self.D.state_dict(), f"disc_ablation_ep{ep}.pt")

if __name__ == "__main__":
    Trainer().train()
