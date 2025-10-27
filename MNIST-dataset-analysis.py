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
import torch.nn.functional as F
import numpy as np
import ot  # from POT

# ----- SETTINGS -----
threshold = 0.8          # pixel threshold in [0,1]
use_sinkhorn = True      # True = faster approximate W1; False = exact EMD
sinkhorn_reg = 5e-2      # entropic regularization (smaller = closer to exact, slower)
device = "cpu"           # keep POT on CPU; torch GPU won’t accelerate POT calls directly

# Build distributions (probability vectors)
def to_prob(img2d, thr=threshold):
    arr = img2d.clone()
    arr = torch.where(arr >= thr, arr, torch.zeros_like(arr))
    mass = arr.sum()
    arr = arr / mass
    return arr.view(-1)

A = torch.stack([to_prob(images[i]) for i in range(N)])  # [N, 28*28]

# Ground cost matrix between pixel coordinates
# Coordinates grid (row, col), Euclidean ground metric
coords = torch.stack(torch.meshgrid(torch.arange(28), torch.arange(28), indexing='ij'), dim=-1).view(-1, 2).float()  # [28*28, 2]
M = torch.cdist(coords, coords, p=2)  # [28*28, 28*28], Euclidean distances
M = M.cpu().numpy().astype(np.float64)  # POT expects numpy

# Convert A to numpy
A_np = A.cpu().numpy().astype(np.float64)  # each row sums to 1

# Pairwise W1 distances
D = np.zeros((N, N), dtype=np.float64)

if use_sinkhorn:
    # Precompute a small epsilon on diagonal for numerical stability if needed
    for i in range(N):
        ai = A_np[i]
        for j in range(i, N):
            bj = A_np[j]
            w2 = ot.sinkhorn2(ai, bj, M, reg=sinkhorn_reg)  # returns transport cost
            w = float(np.array(w2).squeeze())
            D[i, j] = w
            D[j, i] = w
else:
    for i in range(N):
        ai = A_np[i]
        for j in range(i, N):
            bj = A_np[j]
            w = ot.emd2(ai, bj, M)  # exact EMD cost (W1 with our ground metric)
            D[i, j] = w
            D[j, i] = w

# Plot heatmap and draw label-group boundaries
plt.figure(figsize=(7, 7))
im = plt.imshow(D, interpolation='nearest', aspect='auto')
plt.title(f"Wasserstein-1 distance (sorted by label)  N={N}, size={28}x{28}\n"
          + ("Sinkhorn (≈W1, reg={:.3g})".format(sinkhorn_reg) if use_sinkhorn else "Exact EMD"))
plt.colorbar(im, label="W₁")

# boundaries per digit group
counts = [(labels == d).sum().item() for d in range(10)]
cuts = np.cumsum(counts)[:-1]
for c in cuts:
    plt.axhline(c - 0.5, lw=0.5, color='white')
    plt.axvline(c - 0.5, lw=0.5, color='white')

# tick centers
centers = []
start = 0
for cnt in counts:
    centers.append(start + cnt / 2)
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