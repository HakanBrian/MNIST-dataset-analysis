import torch
from torchvision import datasets, transforms
import random

# Define transform to convert images to tensors
transform = transforms.ToTensor()

# Download the MNIST training dataset
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Randomly select 5000 samples
indices = random.sample(range(len(mnist)), 5000)
subset = torch.utils.data.Subset(mnist, indices)

# Take a subset
N = len(subset)
images = torch.stack([img for img, _ in subset]).view(N, -1)  # images in [N, 784]
labels = torch.tensor([label for _, label in subset], dtype=torch.long)  # labels

# Sort by label
order = torch.argsort(labels)
images = images[order]
labels = labels[order]


## Pairwise W1 distances
import numpy as np
from joblib import Parallel, delayed
import ot
from scipy.spatial.distance import cdist

H, W = 28, 28
dtype = np.float64
thr = 0.30
reg = 0.1
num_jobs = -1  # use all cores

# images: torch.Tensor (N,784) in [0,1]
X = images.cpu().numpy().astype(dtype)
N = X.shape[0]
imgs_2d = X.reshape(N, H, W)

def to_distribution(img2d, thr=thr):
    m = (img2d > thr)
    a = m.astype(dtype).reshape(-1)
    if a.sum() == 0:
        a[(H//2)*W + (W//2)] = 1.0
    a = a / a.sum()
    return a

A = np.stack([to_distribution(im) for im in imgs_2d], axis=0)   # (N, 784)

# Ground cost on grid
ys, xs = np.mgrid[0:H, 0:W].reshape(2, -1)
XY = np.stack([ys, xs], axis=1).astype(dtype)
M = cdist(XY, XY, metric='euclidean').astype(dtype)

# scale cost
M /= M.max()              # put distances in [0,1]

# Pick a stabilized solver
def w1_sinkhorn_stable(a, b, M, reg):
    cost = ot.sinkhorn2(a, b, M, reg,
                        method='sinkhorn_stabilized',
                        numItermax=2000, stopThr=1e-7)
    return float(cost)

def w1_row(i):
    ai = A[i]
    row = np.zeros(N, dtype=dtype)
    for j in range(i+1, N):  # upper triangle only
        bj = A[j]
        try:
            d = w1_sinkhorn_stable(ai, bj, M, reg)
        except Exception:
            d = ot.emd2(ai, bj, M)
        row[j] = d
    return i, row

# run rows in parallel, then assemble and symmetrize
pairs = Parallel(n_jobs=num_jobs, prefer="processes", batch_size="auto")(
    delayed(w1_row)(i) for i in range(N)
)

W1 = np.zeros((N, N), dtype=dtype)
for i, row in pairs:
    W1[i, :] = row

iu, ju = np.triu_indices(N, k=1)
W1[ju, iu] = W1[iu, ju]           # mirror to lower triangle

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
im = ax.imshow(W1, interpolation='nearest', aspect='auto', cmap='viridis')
ax.set_title(r"W$_1$")


import matplotlib.pyplot as plt

# colorbar identical style
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Compute counts of each digit (assuming sorted by label)
counts = [(labels == d).sum().item() for d in range(10)]
cuts = torch.tensor(counts).cumsum(0).tolist()[:-1]

# Draw boundaries between digit groups
for c in cuts:
    ax.axhline(c - 0.5, linewidth=0.5, color='white')
    ax.axvline(c - 0.5, linewidth=0.5, color='white')

# Compute tick centers
centers = []
start = 0
for cnt in counts:
    centers.append(start + cnt / 2)
    start += cnt

ax.set_xticks(centers)
ax.set_yticks(centers)
ax.set_xticklabels(list(range(10)))
ax.set_yticklabels(list(range(10)))

ax.set_xlabel("Image Index")
ax.set_ylabel("Image Index")

plt.suptitle("Pairwise W$_1$ Distances (Sorted by Label)")
plt.tight_layout()
plt.savefig("pairwise_dist_W1")
