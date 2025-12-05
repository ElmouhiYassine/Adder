import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# 1) Define your simple CNN
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
#         self.pool  = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.fc1   = nn.Linear(32*7*7, 128)
#         self.fc2   = nn.Linear(128, 10)
#         self.relu  = nn.ReLU()
#
#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         return self.fc2(x)
#
# # 2) Noise injecting transform
# class AddGaussianNoise(object):
#     def __init__(self, mean=0.0, std=0.0):
#         self.mean = mean
#         self.std  = std
#
#     def __call__(self, tensor):
#         if self.std == 0:
#             return tensor
#         img = tensor * 255.0
#         noise = torch.randn_like(img) * self.std + self.mean
#         img_noisy = torch.clamp(img + noise, 0., 255.)
#         return img_noisy / 255.0
#
# # 3) Settings
# batch_size = 64
# train_noise_std = 15.0  # noise level injected during training
#
# # 4) Data loaders with noise in training transforms
# train_transform = transforms.Compose([
#     transforms.ToTensor(),
#     AddGaussianNoise(0.0, train_noise_std)
# ])
#
# train_loader = DataLoader(
#     datasets.FashionMNIST('data', train=True, download=True, transform=train_transform),
#     batch_size=batch_size, shuffle=True
# )
#
# test_clean_loader = DataLoader(
#     datasets.FashionMNIST('data', train=False, download=True, transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=False
# )
#
# # 5) Setup device, model, loss, optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = SimpleCNN().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
#
# # 6) Training loop
# for epoch in range(1, 6):
#     model.train()
#     running_loss = 0.0
#     for imgs, labels in train_loader:
#         imgs, labels = imgs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(imgs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * imgs.size(0)
#     print(f"Epoch {epoch} — train loss: {running_loss / len(train_loader.dataset):.4f}")
#
# # 7) Evaluation function
# def evaluate(loader):
#     model.eval()
#     correct = total = 0
#     with torch.no_grad():
#         for imgs, labels in loader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             preds = model(imgs).argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#     return correct / total
# #
# # 8) Evaluate on clean test set
# acc_clean = evaluate(test_clean_loader)
# print(f"\nAccuracy on CLEAN test set: {acc_clean * 100:.2f}%")
#
# # 9) Evaluate on noisy test sets with different noise stddev
# sigmas = [0, 15, 30, 50, 75, 100]
# print("\nAccuracy on NOISY test sets:")
# for sigma in sigmas:
#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         AddGaussianNoise(0.0, sigma)
#     ])
#     noisy_loader = DataLoader(
#         datasets.FashionMNIST('data', train=False, download=True, transform=test_transform),
#         batch_size=batch_size, shuffle=False
#     )
#     acc = evaluate(noisy_loader)
#     print(f"  σ = {sigma:3d} → Accuracy = {acc * 100:.2f}%")
def plot_noise_example(dataset, idx=0, sigma=30.0, mean=0.0, seed=0, bins=50):
    """
    Shows: [Original]  [Noisy]  [Noise Distribution]
    - dataset must return tensors in [0,1] (e.g., ToTensor()).
    - sigma is the Gaussian stddev in *pixel space* (0..255).
    """
    rng = np.random.default_rng(seed)

    # --- fetch clean image (expects [0,1], shape [1,H,W]) ---
    x, _ = dataset[idx]
    if torch.is_tensor(x):
        x = x.numpy()
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]  # (H,W)

    img255 = (x.astype(np.float32) * 255.0)

    # --- sample raw Gaussian noise in pixel space ---
    noise = rng.normal(loc=mean, scale=sigma, size=img255.shape).astype(np.float32)

    # --- add + clip to valid range ---
    noisy255 = np.clip(img255 + noise, 0.0, 255.0)

    # --- round to uint8 for display (matches common transforms) ---
    orig_disp  = np.rint(img255).astype(np.uint8)
    noisy_disp = np.rint(noisy255).astype(np.uint8)

    # --- plot ---
    fig = plt.figure(figsize=(10.5, 3.4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(orig_disp, cmap='gray', vmin=0, vmax=255)
    ax1.set_title("Original")
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(noisy_disp, cmap='gray', vmin=0, vmax=255)
    ax2.set_title("Noisy")
    ax2.axis('off')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(noise.ravel(), bins=bins)
    ax3.set_title("Noise Distribution")
    ax3.set_xlabel("Noise value")
    ax3.set_ylabel("Frequency")
    ax3.set_xlim(-4*sigma, 4*sigma)  # nice Gaussian window
    ax3.axvline(0, color='k', lw=0.8, ls='--')

    fig.tight_layout()
    return fig

clean_test = datasets.FashionMNIST(
    '../data', train=False, download=True, transform=transforms.ToTensor()
)

_ = plot_noise_example(clean_test, idx=0, sigma=30.0, seed=0)
plt.show()
