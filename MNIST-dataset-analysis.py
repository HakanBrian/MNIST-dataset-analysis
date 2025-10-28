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


## Pairwise distances under norms
import matplotlib.pyplot as plt

# Compute pairwise distances under different norms
dist_L1  = torch.cdist(images, images, p=1)
dist_L2  = torch.cdist(images, images, p=2)
dist_Linf = torch.cdist(images, images, p=float('inf'))

# Plot all three as heatmaps
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = [r"L$_1$", r"L$_2$", r"L$_\infty$"]
dists = [dist_L1, dist_L2, dist_Linf]

for ax, D, title in zip(axes, dists, titles):
    im = ax.imshow(D.cpu().numpy(), interpolation='nearest', aspect='auto', cmap='viridis')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # draw label group boundaries
    counts = [(labels == d).sum().item() for d in range(10)]
    cuts = torch.tensor(counts).cumsum(0).tolist()[:-1]
    for c in cuts:
        ax.axhline(c - 0.5, linewidth=0.5, color='white')
        ax.axvline(c - 0.5, linewidth=0.5, color='white')
    centers = []
    start = 0
    for cnt in counts:
        centers.append(start + cnt / 2)
        start += cnt
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(list(range(10)))
    ax.set_yticklabels(list(range(10)))

plt.suptitle("Pairwise Distances Between MNIST Images (Sorted by Label)")
plt.tight_layout()
plt.savefig("pairwaise_dist_norms.png")


## Pairwise W1 distances
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
import ot

# ----- SETTINGS -----
threshold = 0.8          # pixel threshold in [0,1]
reg = 5e-2               # entropic regularization (smaller => closer to EMD, slower)

# Build distributions (probability vectors)
def to_prob(img2d, thr=threshold):
    arr = img2d.clone()
    arr = torch.where(arr >= thr, arr, torch.zeros_like(arr))
    mass = arr.sum()
    arr = arr / mass
    return arr.view(-1)

# Each row a probability vector that sums to 1
A = torch.stack([to_prob(images[i]) for i in range(N)])  # [N, 28*28]
A_np = A.cpu().numpy().astype(np.float64)  # convert A to numpy

# Ground cost matrix between pixel coordinates
# Coordinates grid (row, col), Euclidean ground metric
coords = torch.stack(torch.meshgrid(torch.arange(28), torch.arange(28), indexing='ij'), dim=-1).view(-1, 2).float()  # [28*28, 2]
M = torch.cdist(coords, coords, p=2)  # [28*28, 28*28], Euclidean distances
M = M.cpu().numpy().astype(np.float64)  # POT expects numpy

# Compute distances for row i for j >= i (upper triangle)
def compute_row(i, A_np, M, reg):
    ai = A_np[i]
    row = np.zeros(A_np.shape[0], dtype=np.float64)
    for j in range(i, A_np.shape[0]):
        bj = A_np[j]
        # sinkhorn2 returns the transport cost <T, M> (≈ W1 for metric M)
        w2 = ot.sinkhorn2(ai, bj, M, reg=reg)
        row[j] = float(np.array(w2).squeeze())
    return i, row

# Parallelize across rows
# inner_max_num_threads=1 prevents BLAS over-subscription per process
with parallel_backend("loky", inner_max_num_threads=1):
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(compute_row)(i, A_np, M, reg) for i in range(N)
    )

# Assemble full symmetric matrix
D = np.zeros((N, N), dtype=np.float64)
for i, row in results:
    D[i, i:] = row[i:]
    D[i:, i] = row[i:]  # mirror to lower triangle

# Plot heatmap
plt.figure(figsize=(7,7))
im = plt.imshow(D, interpolation='nearest', aspect='auto')
plt.title(f"Wasserstein-1 (Sinkhorn ≈, reg={reg}, parallel)")
plt.colorbar(im, label="W₁")

# draw label-group boundaries
counts = [(labels == d).sum().item() for d in range(10)]
cuts = np.cumsum(counts)[:-1]
for c in cuts:
    plt.axhline(c - 0.5, lw=0.5, color='white')
    plt.axvline(c - 0.5, lw=0.5, color='white')

centers, start = [], 0
for cnt in counts:
    centers.append(start + cnt/2)
    start += cnt
plt.xticks(centers, list(range(10)))
plt.yticks(centers, list(range(10)))

plt.tight_layout()
plt.savefig("pairwise_dist_W1")


## SVD
# Get all images of label 7 from our subset
images_7 = [img.view(-1) for img, label in subset if label == 7]
X = torch.stack(images_7, dim=1)  # shape: [784, num_images]

# Compute SVD
U, S, Vt = torch.linalg.svd(X, full_matrices=False)

# Determine the smallest rank with “reasonable” approximation
energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
plt.figure(figsize=(6,4))
plt.plot(energy.numpy(), marker='o')
plt.title("Cumulative explained variance (digit 7)")
plt.xlabel("Rank (k)")
plt.ylabel("Fraction of variance explained")
plt.xlim(0, 100)
plt.grid(True)
plt.savefig("SVD_rank_vs_variance.png")

# Find smallest k with reasonable variance captured
reasonable = 0.90
k95 = torch.where(energy >= reasonable)[0][0].item() + 1
print(f"Smallest rank with ≥{reasonable*100}% variance explained: {k95}")

# Plot low-rank reconstructions
def reconstruct_image(idx=0, k=10):
    # reconstruct image idx using top-k components
    x_hat = (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, idx])
    return x_hat.view(28, 28)

plt.figure(figsize=(10, 3))
for i, k in enumerate([1, 5, 25, 100]):
    plt.subplot(1, 4, i+1)
    plt.imshow(reconstruct_image(0, k), cmap='gray')
    plt.title(f"k = {k}")
    plt.axis('off')
plt.savefig("SVD_lowrank_reconstruction.png")