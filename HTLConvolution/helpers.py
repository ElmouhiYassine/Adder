import numpy as np
import math
from typing import Any


# ========================================================
#                     BASIC HELPERS
# ========================================================

def needed_bt_width(max_abs_sum: int) -> int:
    n = 1
    while (3**n - 1)//2 < max_abs_sum:
        n += 1
    return n

def fit_width(vec, width, pad=0):
    if len(vec) >= width:
        return vec[:width]
    return vec + [pad] * (width - len(vec))

def int_to_binary(n: int, width: int):
    return [+1 if (n >> b) & 1 else -1 for b in range(width)]

def int_to_balanced_ternary(n: int, min_len: int = None):
    if n == 0:
        vec = [0]
    else:
        vec = []
        num = n
        while num != 0:
            r = num % 3
            if r == 0:
                vec.append(0)
                num //= 3
            elif r == 1:
                vec.append(+1)
                num = (num - 1) // 3
            elif r == 2:
                vec.append(-1)
                num = (num + 1) // 3
    if min_len is not None and len(vec) < min_len:
        vec += [0] * (min_len - len(vec))
    return vec

def bits_to_int(lsb_vec: list[int]) -> int:
    val = 0
    for b, t in enumerate(lsb_vec):
        if t == +1:
            val |= (1 << b)
    return val

def int_to_bits(n: int, width: int) -> list[int]:
    return [+1 if (n >> b) & 1 else -1 for b in range(width)]


# ========================================================
#                 UNCERTAINTY MASKING
# ========================================================

def uncertainty_mask_threshold(noise_abs: int, K1: int):
    mask = np.ones(K1, dtype=bool)
    b = 0
    while b < K1 and (2**b) <= noise_abs:
        mask[b] = False
        b += 1
    return mask

def uncertainty_mask_even(noise_abs: int, K1: int):
    mask = np.ones(K1, dtype=bool)
    if noise_abs <= 0:
        return mask
    msb = int(math.floor(math.log2(noise_abs)))
    for b in range(0, min(K1, msb + 1), 2):
        mask[b] = False
    return mask

def uncertainty_mask_bitset(noise_abs: int, K1: int):
    mask = np.ones(K1, dtype=bool)
    for b in range(K1):
        if (noise_abs >> b) & 1:
            mask[b] = False
    return mask


# ========================================================
#                      ENCODING
# ========================================================

def encode_uint8_to_UV(img_noisy: np.ndarray,
                       noise_abs_map: np.ndarray,
                       K1: int,
                       K2: int,
                       uncertainty_policy: str = "threshold"):

    H, W = img_noisy.shape
    U = np.zeros((H, W, K1), dtype=int)
    V = np.zeros((H, W, K2), dtype=int)

    if uncertainty_policy == "threshold":
        mask_fn = uncertainty_mask_threshold
    elif uncertainty_policy == "even":
        mask_fn = uncertainty_mask_even
    elif uncertainty_policy == "bitset":
        mask_fn = uncertainty_mask_bitset
    else:
        raise ValueError(f"Unknown uncertainty_policy: {uncertainty_policy}")

    for i in range(H):
        for j in range(W):
            p = int(img_noisy[i, j])
            nz = int(noise_abs_map[i, j])

            mask = mask_fn(nz, K1)

            for b in range(K1):
                if not mask[b]:
                    U[i, j, b] = 0
                else:
                    U[i, j, b] = (+1 if ((p >> b) & 1) else -1)

            upper_abs = (p >> K1) << K1
            V[i, j, :] = fit_width(int_to_balanced_ternary(upper_abs), K2, 0)

    return np.concatenate([U, V], axis=2), U, V


# ========================================================
#                     DECODING
# ========================================================

def phi_center(t):
    return 0.0 if t == -1 else (0.5 if t == 0 else 1.0)

def decode_U_center(u_lsb: np.ndarray) -> float:
    val = 0.0
    for i, d in enumerate(u_lsb.tolist()):
        val += phi_center(d) * (2 ** i)
    return val

def decode_V_integer(v_bt_lsb: np.ndarray) -> int:
    total = 0
    for i, d in enumerate(v_bt_lsb.tolist()):
        total += int(d) * (3 ** i)
    return total

def decode_Y_numeric_center(Y: np.ndarray, K1: int, K2: int) -> np.ndarray:
    H, W, _ = Y.shape
    out = np.zeros((H, W), dtype=float)
    for i in range(H):
        for j in range(W):
            u = Y[i, j, :K1]
            v = Y[i, j, K1:K1+K2]
            out[i, j] = decode_U_center(u) + decode_V_integer(v)
    return out


# ========================================================
#           CLASSIC BASELINE FOR COMPARISON
# ========================================================

def conv2d_valid_int(img: np.ndarray, ker: np.ndarray) -> np.ndarray:
    H, W = img.shape
    k = ker.shape[0]
    out = np.zeros((H - k + 1, W - k + 1), dtype=int)
    for i in range(H - k + 1):
        for j in range(W - k + 1):
            out[i, j] = int(np.sum(img[i:i+k, j:j+k] * ker))
    return out


# ========================================================
#                     METRICS
# ========================================================

def psnr(mse, max_val=255.0):
    return float("inf") if mse == 0 else 10.0 * math.log10((max_val**2) / mse)


# ========================================================
#                 EXPERIMENT RUNNER
# ========================================================

from HTLConvolution.dataset_loader import load_xmnist_first_N
from HTLConvolution.htl_convolution import convolution

def run_mnist_noise_experiment(n_images: int = 32,
                               K1: int = 4,
                               K2: int = 4,
                               kernel: np.ndarray = None,
                               noise_sigma: float = 5.0,
                               noise_dist: str = "gaussian",
                               uncertainty_policy: str = "threshold",
                               dataset: str = "fashion",
                               seed: int = 1234):

    rng = np.random.default_rng(seed)
    imgs = load_xmnist_first_N(dataset=dataset, n_samples=n_images)

    if kernel is None:
        kernel = np.ones((3, 3), dtype=int)

    k = kernel.shape[0]
    mae_naive = []
    mse_naive = []
    mae_fw = []
    mse_fw = []

    for idx in range(imgs.shape[0]):
        img = imgs[idx].astype(int)

        # --- noise injection ---
        if noise_dist == "gaussian":
            noise = rng.normal(0, noise_sigma, size=img.shape)
        elif noise_dist == "uniform":
            noise = rng.uniform(-noise_sigma, noise_sigma, size=img.shape)
        else:
            raise ValueError("Invalid noise_dist")

        noise = noise.round().astype(int)
        noise_abs = np.abs(noise)
        img_noisy = np.clip(img + noise, 0, 255).astype(int)

        # clean conv
        base = conv2d_valid_int(img, kernel)

        # naive conv
        naive = conv2d_valid_int(img_noisy, kernel)

        # our framework
        X, U, V = encode_uint8_to_UV(img_noisy, noise_abs, K1, K2,
                                     uncertainty_policy)

        from Adders.SK_Quasi_adder import ripple_quasi
        from Adders.balanced_ternary_adder import balanced_ternary_add

        Y = convolution(X, kernel, ripple_quasi, balanced_ternary_add,
                        K1=K1, K2=K2)
        fw_map = decode_Y_numeric_center(Y, K1, K2)

        mae_naive.append(np.mean(np.abs(naive - base)))
        mse_naive.append(np.mean((naive - base) ** 2))
        mae_fw.append(np.mean(np.abs(fw_map - base)))
        mse_fw.append(np.mean((fw_map - base) ** 2))

    return {
        "noise_sigma": float(noise_sigma),
        "mae_naive": float(np.mean(mae_naive)),
        "mse_naive": float(np.mean(mse_naive)),
        "mae_fw": float(np.mean(mae_fw)),
        "mse_fw": float(np.mean(mse_fw)),
    }
