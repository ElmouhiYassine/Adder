import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from preprocessing_unanimous import unanimous_impute

# ── Step 1: Load MNIST once ──────────────────────────────────────────────
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
ALL_IMAGES = mnist.data.reshape(-1, 28, 28)    # shape = (70000, 28, 28)
ALL_LABELS = mnist.target.astype(int)          # shape = (70000,)

INDICES_BY_DIGIT = {d: np.where(ALL_LABELS == d)[0] for d in range(10)}


def select_image(digit=2, seed=None):
    if seed is not None:
        np.random.seed(seed)

    idx_list = INDICES_BY_DIGIT[digit]
    pick = np.random.choice(idx_list)
    return ALL_IMAGES[pick]


def binarize(image, thresh=128):
    return (image > thresh).astype(int)

def ternarize(bin_img):
    return np.where(bin_img == 1, 1, -1)

def add_noise(tern_img, p=0.1, seed=42):
    if seed is not None:
        np.random.seed(seed)
    H, W = tern_img.shape
    total = H*W
    n_flip = int(np.floor(p*total))
    indices = np.random.choice(total, size=n_flip, replace=False)
    noisy = tern_img.copy()
    noisy.flat[indices] = 0
    mask = np.zeros_like(tern_img, dtype=bool)
    mask.flat[indices] = True
    return noisy, mask

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


    return noisy

def majority_impute(noisy, kernel_size=3):
    H, W = noisy.shape
    pad = kernel_size // 2
    padded = np.pad(noisy, pad, mode='constant', constant_values=0)
    imputed = noisy.copy()
    for i in range(H):
        for j in range(W):
            if noisy[i, j] != 0:
                continue
            win = padded[i:i+kernel_size, j:j+kernel_size].copy()
            win[pad, pad] = 0
            pos = np.count_nonzero(win == 1)
            neg = np.count_nonzero(win == -1)
            if pos > neg:
                imputed[i, j] = 1
            elif neg > pos:
                imputed[i, j] = -1
    return imputed

def add_stripe_noise(tern_img, stripe_width=2):
    H, W = tern_img.shape
    noisy = tern_img.copy()
    mask = np.zeros_like(tern_img, dtype=bool)

    for i in range(0, H, 2 * stripe_width):
        noisy[i:i + stripe_width, :] = 0
        mask[i:i + stripe_width, :] = True

    return noisy, mask


# === Visualization ===
def visualize_imputation(orig, noisy, imputed):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Make figure wider and taller
    imgs = [orig, noisy, imputed]
    titles = ["Original Ternary", "Noisy Ternary", "Imputed Ternary"]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap='hot', vmin=-1, vmax=1)
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    plt.subplots_adjust(top=0.95, wspace=0.3)  # Add more top space
    plt.show()

def evaluate_imputation(n_runs=20, p=0.3, kernel_size=3, seed=None):
    accuracies = []
    digits = np.arange(10)

    if seed is not None:
        np.random.seed(seed)

    for _ in range(n_runs):
        digit = np.random.choice(digits)
        img = select_image(digit)
        bin_img = binarize(img)
        ter = ternarize(bin_img)

        noisy, mask = add_noise(ter, p=p)
        imputed = unanimous_impute(noisy, kernel_size=kernel_size)

        flipped_positions = np.where(mask)
        n_flipped = mask.sum()
        if n_flipped == 0:
            continue  # skip if no noise was added (edge case)

        correct = sum(imputed[i, j] == ter[i, j] for i, j in zip(*flipped_positions))
        accuracy = correct / n_flipped
        accuracies.append(accuracy)

    if not accuracies:
        return 0.0, 0

    average_accuracy = np.mean(accuracies)
    return average_accuracy, len(accuracies)



# === Run and visualize ===
if __name__ == "__main__" :
    avg_acc, runs = evaluate_imputation(n_runs=1000, p=0.5, kernel_size=3, seed=42)
    print(f"Average Imputation Accuracy over {runs} runs: {avg_acc * 100:.2f}%")









    # digit = 7
    # img = select_image(digit)
    # bin_img = binarize(img)
    # ter = ternarize(bin_img)
    # noisy, mask = add_stripe_noise(ter,stripe_width=3)
    # imputed = majority_impute(noisy, kernel_size=3)
    # visualize_imputation(ter, noisy, imputed)
    # flipped_positions = np.where(mask)
    # n_flipped = mask.sum()
    # # Count how many unknowns were imputed (i.e., changed from 0)
    # imputed_mask = (noisy == 0) & (imputed != 0)
    # n_imputed = imputed_mask.sum()
    # correct = 0
    # for i, j in zip(*flipped_positions):
    #     if imputed[i, j] == ter[i, j]:
    #         correct += 1
    # p = 0.7
    # accuracy = correct / n_flipped if n_flipped > 0 else 0.0
    # # print(f"Noise fraction p={p:.2f}, flipped pixels = {n_flipped}")
    # print(f"flipped pixels = {n_flipped}")
    # print(f"Total imputed pixels: {n_imputed}")
    # print(f"Imputation accuracy = {accuracy * 100:.1f}%")