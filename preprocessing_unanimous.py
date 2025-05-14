import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


# === Helper functions ===
def select_image(digit=2):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    imgs = mnist.data.reshape(-1, 28, 28)
    labs = mnist.target.astype(int)
    return imgs[labs == digit][4]


def binarize(image, thresh=128):
    return (image > thresh).astype(int)


def ternarize(bin_img):
    return np.where(bin_img == 1, 1, -1)


def add_noise(tern_img, p=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    H, W = tern_img.shape
    total = H * W
    n_flip = int(np.floor(p * total))
    indices = np.random.choice(total, size=n_flip, replace=False)
    noisy = tern_img.copy()
    noisy.flat[indices] = 0
    mask = np.zeros_like(tern_img, dtype=bool)
    mask.flat[indices] = True
    return noisy, mask


def unanimous_impute(noisy, kernel_size=3):
    """
    Impute 0→1 only if all non-zero neighbors are 1,
    0→-1 only if all non-zero neighbors are -1.
    Otherwise leave 0.
    """
    H, W = noisy.shape
    pad = kernel_size // 2
    padded = np.pad(noisy, pad, mode='constant', constant_values=0)
    imputed = noisy.copy()
    for i in range(H):
        for j in range(W):
            if noisy[i, j] != 0:
                continue
            win = padded[i:i + kernel_size, j:j + kernel_size].copy().flatten()
            win = win[win != 0]  # non-zero neighbors
            if win.size == 0:
                continue
            if np.all(win == 1):
                imputed[i, j] = 1
            elif np.all(win == -1):
                imputed[i, j] = -1
    return imputed

def add_block_noise(tern_img, block_size=10, n_blocks=1, seed=None):
    """Insert n_blocks random square unknown blocks of size block_size."""
    if seed is not None:
        np.random.seed(seed)

    H, W = tern_img.shape
    noisy = tern_img.copy()
    mask = np.zeros_like(tern_img, dtype=bool)

    for _ in range(n_blocks):
        i = np.random.randint(0, H - block_size + 1)
        j = np.random.randint(0, W - block_size + 1)
        noisy[i:i + block_size, j:j + block_size] = 0
        mask[i:i + block_size, j:j + block_size] = True

    return noisy, mask

# === Visualization and evaluation ===
def visualize_and_evaluate(p=0.1, kernel_size=3):
    # Load and prepare
    img = select_image(digit=8)
    bin_img = binarize(img)
    ter = ternarize(bin_img)

    # Add noise
    noisy, mask = add_block_noise(ter,block_size=4,n_blocks=15,seed = 42)

    # Impute
    imputed = unanimous_impute(noisy, kernel_size=kernel_size)

    # Evaluate accuracy on flipped pixels
    flipped_positions = np.where(mask)
    n_flipped = mask.sum()
    correct = (imputed[flipped_positions] == ter[flipped_positions]).sum()
    accuracy = correct / n_flipped if n_flipped > 0 else 0.0

    # Count how many unknowns were imputed (i.e., changed from 0)
    imputed_mask = (noisy == 0) & (imputed != 0)
    n_imputed = imputed_mask.sum()

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Make figure wider and taller
    imgs = [ter, noisy, imputed]
    titles = ["Original Ternary", "Noisy Ternary", "Imputed Ternary"]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap='hot', vmin=-1, vmax=1)
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    plt.subplots_adjust(top=0.95, wspace=0.3)  # Add more top space
    plt.show()

    print(f"Flipped pixels: {n_flipped}")
    print(f"imputed pixels: {n_imputed}")
    print(f"Unanimous Imputation accuracy: {accuracy * 100:.1f}%")


# Run
visualize_and_evaluate(p=0.5, kernel_size=3)
