#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graph-conditioned GAN with Weights & Biases logging:
• super-piksele SLIC → RAG
• GAT-encoder → embedding grafu (z_graph)
• z_graph + z_noise → CNN-generator → 256×256 RGB
• CNN-dyskryminator (RGB) + BCE + R1
• logowanie strat, metryk i przykładowych obrazów do wandb
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
import torchvision.transforms as T
import torchvision.utils as vutils

import wandb

from skimage.segmentation import slic
from skimage import graph as skgraph

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool

# ==================== CONFIG ====================
class Cfg:
    dataset_path   = "/home/student2/histo/data/lung_colon_image_set/lung_image_sets/lung_scc"
    image_size     = 256
    batch_size     = 8
    num_workers    = 4
    z_dim          = 128
    gat_dim        = 128
    gat_heads      = 4
    gat_layers     = 3
    gen_feat       = 512      # kanały w 4×4 tensorze startowym
    d_feat         = 64
    lr             = 2e-4
    beta1, beta2   = 0.0, 0.99
    r1_gamma       = 10.0
    num_epochs     = 200
    num_spixels    = 200
    log_every      = 100      # log co tyle kroków

cfg = Cfg()

# ==================== WEIGHTS & BIASES ====================
wandb.init(
    project="graph_gan",
    config={
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "beta1": cfg.beta1,
        "beta2": cfg.beta2,
        "z_dim": cfg.z_dim,
        "gat_dim": cfg.gat_dim,
        "num_spixels": cfg.num_spixels,
        "gen_feat": cfg.gen_feat,
        "d_feat": cfg.d_feat,
    }
)
wandb.config.update({"image_size": cfg.image_size, "epochs": cfg.num_epochs})

# ==================== DATASET ====================
class GraphImageDataset(Dataset):
    def __init__(self, root):
        root = Path(root)
        exts = {".png", ".jpg", ".jpeg"}
        self.paths = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")
        self.tf_img = T.Compose([
            T.Resize((cfg.image_size, cfg.image_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.paths)

    def _img_to_graph(self, img_np):
        spx = slic(img_np, n_segments=cfg.num_spixels, compactness=10, start_label=0)
        n_nodes = spx.max() + 1
        feats = np.zeros((n_nodes, 3), dtype=np.float32)
        counts = np.bincount(spx.flatten())
        for c in range(3):
            vals = np.bincount(spx.flatten(), weights=img_np[..., c].flatten())
            feats[:, c] = vals / counts
        rag = skgraph.rag_mean_color(img_np, spx)
        edges = np.array([[u, v] for u, v in rag.edges()], dtype=np.int64)
        if edges.size == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges.T, dtype=torch.long)
        return Data(x=torch.tensor(feats), edge_index=edge_index)

    def __getitem__(self, idx):
        p = self.paths[idx]
        pil = Image.open(p).convert("RGB")
        arr = np.asarray(pil.resize((cfg.image_size, cfg.image_size))) / 255.0
        graph = self._img_to_graph(arr)
        img_t = self.tf_img(pil)
        return img_t, graph


def collate(batch):
    imgs, graphs = zip(*batch)
    return torch.stack(imgs), Batch.from_data_list(graphs)

# ==================== GRAPH ENCODER ====================
class GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 3
        self.convs = nn.ModuleList()
        for _ in range(cfg.gat_layers):
            self.convs.append(GATv2Conv(in_dim, cfg.gat_dim, heads=cfg.gat_heads, concat=False))
            in_dim = cfg.gat_dim
        self.lin = nn.Linear(cfg.gat_dim, cfg.z_dim)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        g = global_mean_pool(x, batch)
        return self.lin(g)

# ==================== GENERATOR ====================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_size = 4
        n_up = int(math.log2(cfg.image_size / self.init_size))
        self.proj = nn.Linear(cfg.z_dim*2, cfg.gen_feat * self.init_size**2)
        feats = [cfg.gen_feat // (2**i) for i in range(n_up+1)]
        self.ups = nn.ModuleList()
        for i in range(n_up):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(feats[i], feats[i+1], 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(feats[i+1], feats[i+1], 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            ))
        self.to_rgb = nn.Conv2d(feats[-1], 3, 1)

    def forward(self, z_graph, B):
        z_noise = torch.randn(B, cfg.z_dim, device=z_graph.device)
        z = torch.cat([z_graph, z_noise], dim=1)
        x = self.proj(z).view(B, cfg.gen_feat, self.init_size, self.init_size)
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
        self.dev = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.E = GraphEncoder().to(self.dev)
        self.G = Generator().to(self.dev)
        self.D = Discriminator().to(self.dev)

        # osobne optymalizatory dla encoder+generator oraz discriminator
        self.optE = torch.optim.Adam(
            self.E.parameters(),
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
        )
        self.optG = torch.optim.Adam(
            self.G.parameters(),
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
        )
        self.optD = torch.optim.Adam(
            self.D.parameters(),
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
        )

        ds = GraphImageDataset(cfg.dataset_path)
        self.loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate,
            pin_memory=True
        )
        self.bce = nn.BCEWithLogitsLoss()

    def r1_penalty(self, real_img, real_pred):
        # real_pred: predykcje D dla rzeczywistych obrazów
        (grad,) = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_img,
            create_graph=True
        )
        return grad.flatten(1).pow(2).sum(1).mean()

    def train(self):
        step = 0
        for ep in range(cfg.num_epochs):
            pbar = tqdm(self.loader, desc=f"Epoch {ep}")
            for imgs, graphs in pbar:
                B = imgs.size(0)
                imgs, graphs = imgs.to(self.dev), graphs.to(self.dev)

                # — Discriminator step —
                self.optD.zero_grad(set_to_none=True)
                imgs.requires_grad_()

                # real
                r_log  = self.D(imgs)
                r_loss = self.bce(r_log, torch.ones_like(r_log))

                # fake (bez gradów dla E/G)
                with torch.no_grad():
                    zg     = self.E(graphs)
                    f_imgs = self.G(zg, B)
                f_log  = self.D(f_imgs)
                f_loss = self.bce(f_log, torch.zeros_like(f_log))

                # LICZYMY R1 TYLKO RAZ
                r1     = self.r1_penalty(imgs, r_log)

                # strata D
                d_loss = 0.5 * (r_loss + f_loss) + 0.5 * cfg.r1_gamma * r1
                d_loss.backward()
                self.optD.step()

                # — Generator+Encoder step —
                self.optE.zero_grad(set_to_none=True)
                self.optG.zero_grad(set_to_none=True)

                zg     = self.E(graphs)
                f_imgs = self.G(zg, B)
                g_log  = self.D(f_imgs)
                g_loss = self.bce(g_log, torch.ones_like(g_log))

                g_loss.backward()
                self.optG.step()
                self.optE.step()

                # — Logowanie do wandb co cfg.log_every kroków —
                if step % cfg.log_every == 0:
                    img_grid = vutils.make_grid(f_imgs.detach(), normalize=True, scale_each=True)
                    wandb.log({
                        "d_loss":    d_loss.item(),
                        "g_loss":    g_loss.item(),
                        "r1_penalty": r1.item(),
                        "generated_images": [wandb.Image(img_grid, caption=f"step_{step}")],
                        "epoch":     ep,
                        "step":      step
                    }, step=step)

                step += 1

            # Checkpoint co 10 epok
            if ep % 10 == 0:
                torch.save(self.E.state_dict(), f"enc_ep{ep}.pt")
                torch.save(self.G.state_dict(), f"gen_ep{ep}.pt")
                torch.save(self.D.state_dict(), f"disc_ep{ep}.pt")

if __name__ == "__main__":
    Trainer().train()
