#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graph-conditioned GAN on CIFAR-10 with superpixel graphs
"""
import math
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from torchvision.datasets import CIFAR10

import wandb
from skimage.segmentation import slic
from skimage import graph as skgraph
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool

# ==================== CONFIG ====================
class Cfg:
    root           = "./data"
    image_size     = 64       # rozdzielczość wejściowa po upsamplingu
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
    num_spixels    = 100      # dla CIFAR-10 wystarczy ~100
    compactness    = 10.0
    log_every      = 100

cfg = Cfg()

# ==================== WEIGHTS & BIASES ====================
wandb.init(
    project="graph_gan_cifar",
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
def collate_fn(batch):
    imgs, graphs = zip(*batch)
    return torch.stack(imgs), Batch.from_data_list(graphs)

class GraphImageDataset(Dataset):
    """
    CIFAR-10 → superpixels → RAG → graph
    Returns: image tensor [-1,1], and torch_geometric.Data graph
    """
    def __init__(self, root, train=True,
                 image_size=64, num_spixels=100, compactness=10.0):
        super().__init__()
        self.cifar = CIFAR10(root=root, train=train, download=True)
        self.image_size = image_size
        self.num_spixels = num_spixels
        self.compactness = compactness
        self.tf_img = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.cifar)

    def _img_to_graph(self, img_np):
        spx = slic(img_np, n_segments=self.num_spixels,
                   compactness=self.compactness, start_label=0)
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
        x = torch.tensor(feats)
        return Data(x=x, edge_index=edge_index)

    def __getitem__(self, idx):
        img, _ = self.cifar[idx]
        pil_resized = img.resize((self.image_size, self.image_size))
        img_np = np.asarray(pil_resized, dtype=np.float32) / 255.0
        graph = self._img_to_graph(img_np)
        img_t = self.tf_img(img)
        return img_t, graph

# ==================== GRAPH ENCODER ====================
class GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 3
        self.convs = nn.ModuleList()
        for _ in range(cfg.gat_layers):
            self.convs.append(GATv2Conv(in_dim, cfg.gat_dim,
                                        heads=cfg.gat_heads, concat=False))
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
        self.proj = nn.Linear(cfg.z_dim*2,
                              cfg.gen_feat * self.init_size**2)
        feats = [cfg.gen_feat // (2**i) for i in range(n_up+1)]
        self.ups = nn.ModuleList()
        for i in range(n_up):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(feats[i], feats[i+1], 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(feats[i+1], feats[i+1], 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ))
        self.to_rgb = nn.Conv2d(feats[-1], 3, 1)

    def forward(self, z_graph, B):
        z_noise = torch.randn(B, cfg.z_dim, device=z_graph.device)
        z = torch.cat([z_graph, z_noise], dim=1)
        x = self.proj(z).view(B, cfg.gen_feat,
                               self.init_size, self.init_size)
        for up in self.ups:
            x = up(x)
        return torch.tanh(self.to_rgb(x))

# ==================== DISCRIMINATOR ====================
class Discriminator(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        layers = []
        in_ch = 3
        feats = [cfg.d_feat, cfg.d_feat*2,
                 cfg.d_feat*4, cfg.d_feat*8]
        for f in feats:
            layers += [nn.Conv2d(in_ch, f, 4, 2, 1),
                       nn.LeakyReLU(0.2, inplace=True)]
            in_ch = f
        self.net = nn.Sequential(*layers)
        n_down = len(feats)
        final_sz = image_size // (2**n_down)
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
        self.E = GraphEncoder().to(self.dev)
        self.G = Generator().to(self.dev)
        self.D = Discriminator(cfg.image_size).to(self.dev)
        self.optE = torch.optim.Adam(self.E.parameters(),
                                     lr=cfg.lr,
                                     betas=(cfg.beta1, cfg.beta2))
        self.optG = torch.optim.Adam(self.G.parameters(),
                                     lr=cfg.lr,
                                     betas=(cfg.beta1, cfg.beta2))
        self.optD = torch.optim.Adam(self.D.parameters(),
                                     lr=cfg.lr,
                                     betas=(cfg.beta1, cfg.beta2))

        ds = GraphImageDataset(root=cfg.root,
                              train=True,
                              image_size=cfg.image_size,
                              num_spixels=cfg.num_spixels,
                              compactness=cfg.compactness)
        self.loader = DataLoader(ds,
                                 batch_size=cfg.batch_size,
                                 shuffle=True,
                                 num_workers=cfg.num_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=True)
        self.bce = nn.BCEWithLogitsLoss()

    def r1_penalty(self, real_img, real_pred):
        (grad,) = torch.autograd.grad(outputs=real_pred.sum(),
                                      inputs=real_img,
                                      create_graph=True)
        return grad.flatten(1).pow(2).sum(1).mean()

    def train(self):
        step = 0
        for ep in range(cfg.num_epochs):
            for imgs, graphs in tqdm(self.loader, desc=f"Epoch {ep}"):
                B = imgs.size(0)
                imgs, graphs = imgs.to(self.dev), graphs.to(self.dev)
                # -- Discriminator --
                self.optD.zero_grad()
                imgs.requires_grad_()
                r_log = self.D(imgs)
                r_loss = self.bce(r_log, torch.ones_like(r_log))
                with torch.no_grad():
                    zg = self.E(graphs)
                    f_imgs = self.G(zg, B)
                f_log = self.D(f_imgs)
                f_loss = self.bce(f_log, torch.zeros_like(f_log))
                r1 = self.r1_penalty(imgs, r_log)
                d_loss = 0.5*(r_loss+f_loss) + 0.5*cfg.r1_gamma*r1
                d_loss.backward()
                self.optD.step()
                # -- Generator & Encoder --
                self.optE.zero_grad()
                self.optG.zero_grad()
                zg = self.E(graphs)
                f_imgs = self.G(zg, B)
                g_log = self.D(f_imgs)
                g_loss = self.bce(g_log, torch.ones_like(g_log))
                g_loss.backward()
                self.optG.step()
                self.optE.step()
                # -- WandB Logging --
                if step % cfg.log_every == 0:
                    grid = vutils.make_grid(f_imgs.detach(),
                                            normalize=True,
                                            scale_each=True)
                    wandb.log({
                        "d_loss": d_loss.item(),
                        "g_loss": g_loss.item(),
                        "r1_penalty": r1.item(),
                        "generated_images": [wandb.Image(grid)],
                        "step": step,
                        "epoch": ep
                    }, step=step)
                step += 1
            if ep % 10 == 0:
                torch.save(self.E.state_dict(), f"enc_ep{ep}.pt")
                torch.save(self.G.state_dict(), f"gen_ep{ep}.pt")
                torch.save(self.D.state_dict(), f"disc_ep{ep}.pt")

if __name__ == "__main__":
    Trainer().train()