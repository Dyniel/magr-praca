import os, glob
import numpy as np
import torch
from skimage.segmentation import slic
from skimage.util import img_as_float
from PIL import Image
from torchvision import transforms

# Ścieżki
dataset_path = "/home/student2/histo/data/lung_colon_image_set/lung_image_sets/lung_scc"
out_dir      = "/home/student2/histo/superpixel_cache"
os.makedirs(out_dir, exist_ok=True)

# Parametry superpikseli
num_superpixels = 300
compactness     = 10

# Transformacja obrazu do 224×224 (jak w treningu)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

for img_path in glob.glob(os.path.join(dataset_path, "*.jpeg")) + \
                glob.glob(os.path.join(dataset_path, "*.jpg"))  + \
                glob.glob(os.path.join(dataset_path, "*.png")):
    base = os.path.splitext(os.path.basename(img_path))[0]
    cache_p = os.path.join(out_dir, f"{base}.pt")
    if os.path.exists(cache_p):
        continue

    # 1) wczytaj i przeskaluj
    img = Image.open(img_path).convert("RGB")
    x   = transform(img)                             # [3,H,W]
    im  = img_as_float(x.permute(1,2,0).numpy())      # [H,W,3]

    # 2) segmentacja
    seg = slic(im, n_segments=num_superpixels,
               compactness=compactness, start_label=0)
    seg[seg>=num_superpixels] = 0

    # 3) cechy: średnie kolory
    feats = np.zeros((num_superpixels, 3), np.float32)
    for s in range(num_superpixels):
        mask = seg==s
        if mask.sum()>0:
            feats[s] = im[mask].mean(axis=0)

    # 4) adjacency
    H,W = seg.shape
    A   = np.eye(num_superpixels, dtype=np.float32)
    for r in range(H):
        for c in range(W):
            cur = seg[r,c]
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr,cc = r+dr, c+dc
                if 0<=rr<H and 0<=cc<W:
                    nei = seg[rr,cc]
                    if nei!=cur:
                        A[cur,nei] = 1.0
    D    = A.sum(axis=1)
    Dinv = np.diag(1.0/np.sqrt(D+1e-6))
    Ahat = Dinv @ A @ Dinv

    # 5) zapisz tensor
    torch.save({
        "feats": torch.from_numpy(feats),
        "A":     torch.from_numpy(Ahat)
    }, cache_p)

    print(f"Cached superpixels for {base}")