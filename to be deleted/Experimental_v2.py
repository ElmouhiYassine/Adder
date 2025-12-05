import numpy as np
from Adders.SK_Quasi_adder import map_quasi_adder
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms

def int_to_8bit_array(val):
    return [(val >> i) & 1 for i in reversed(range(8))]

def encode_pixel(val, lsb_mask):
    """
    Encode one pixel given its 4-bit LSB uncertainty mask.
    - val: 0–255 integer
    - lsb_mask: list/array of length 4 (bool), True means uncertain → trit=0
    Returns 1D np.array of length 8 (LSB→MSB) in {-1,0,1}.
    """
    bits = int_to_8bit_array(val)[::-1]        # LSB→MSB
    trits = [1 if b else -1 for b in bits]     # map 1→+1, 0→−1
    for k in range(4):                         # apply uncertainty mask
        if lsb_mask[k]:
            trits[k] = 0
    return np.array(trits, dtype=int)

def encode_image_with_uncertainty(image, sigma):
    """
    image: 2D np.uint8 array (values 0–255)
    sigma: standard deviation of Gaussian noise
    Returns: 3D np.array of shape (H, W, 8) of ternary vectors
    """
    H, W = image.shape
    # 1) Sample noise per pixel
    noise = np.random.normal(loc=0.0, scale=sigma, size=(H, W))
    noise_abs = np.abs(noise)

    # 2) Build LSB masks: for each bit k=0..3, uncertain if |noise| >= 2^k
    #    mask[:, :, k] = True/False
    thresholds = np.array([2**k for k in range(4)])  # [1,2,4,8]
    # broadcast: (H,W,1) >= (4,) → (H,W,4)
    lsb_mask = noise_abs[:, :, None] >= thresholds[None, None, :]

    # 3) Encode every pixel
    encoded = np.zeros((H, W, 8), dtype=int)
    for i in range(H):
        for j in range(W):
            encoded[i, j] = encode_pixel(int(image[i, j]), lsb_mask[i, j])

    return encoded


def vector_ripple_add(vecs, adder, bits=8):
    """
    Ripple-carry addition of a list of B-bit ternary vectors.
    Uses a single carry value across all vectors and bit positions,
    then appends the final carry if it's not the neutral (-1).

    Args:
      vecs:  list of np.ndarray shape (bits,), values in {-1,0,1}
      adder: function(a, b, cin) -> (sum_trit, carry_trit)
      bits:  base number of trits per vector (e.g. 8)

    Returns:
      np.ndarray of shape (bits,) or (bits+1,)
    """
    # Start with first vector
    result = vecs[0].copy()
    carry  = -1  # neutral carry

    # Fold in all others
    for vec in vecs[1:]:
        for i in range(bits):
            s, carry = adder(int(result[i]), int(vec[i]), carry)
            result[i] = s

    # Append final carry if meaningful
    if carry != -1:
        result = np.concatenate([result, np.array([carry], dtype=int)])
    return result

def map_vector_to_trit(vec, threshold):
    """
    Map an N-bit ternary vector to a single trit:
      - If any bit == 0 -> return 0 (uncertain)
      - Else map each trit: -1->0, +1->1, interpret as unsigned integer
      - If value >= threshold -> +1 else -> -1

    Args:
      vec:       np.ndarray, values in {-1,0,1}
      threshold: int in [0 .. 2^len(vec)-1]
    """
    # 1) Uncertain propagation
    if np.any(vec == 0):
        return 0

    # 2) Build binary bits
    #    -1 -> 0, +1 -> 1
    bin_bits = ((vec + 1) // 2).astype(int)
    # LSB is bin_bits[0], MSB is bin_bits[-1]
    value = sum(bit << i for i, bit in enumerate(bin_bits))

    # 3) Threshold to ±1
    return 1 if value >= threshold else -1

def ternary_convolution(feature_map, kernels,
                        adder=map_quasi_adder,
                        threshold=None):
    """
    Perform ternary convolution over a B‑bit feature map.

    Args:
      feature_map: np.ndarray, shape (H, W, C, B), values in {-1,0,1}
      kernels:     np.ndarray, shape (M, C, k, k), values in {-1,0,1}
      adder:       full-adder function(a,b,cin)
      threshold:   optional int threshold; default = 2^(B) // 2

    Returns:
      output: np.ndarray, shape (H-k+1, W-k+1, M), values in {-1,0,1}
    """
    H, W, C, B = feature_map.shape
    M, _, k, _ = kernels.shape
    out_h, out_w = H - k + 1, W - k + 1

    # Default threshold: halfway point of B‑bit range
    if threshold is None:
        threshold = (1 << B) // 2

    output = np.zeros((out_h, out_w, M), dtype=int)

    for m in range(M):
        kern = kernels[m]  # (C, k, k)
        for i in range(out_h):
            for j in range(out_w):
                vecs = []
                # Gather all active B‑bit vectors multiplied by ±1
                for c in range(C):
                    patch = feature_map[i:i+k, j:j+k, c]  # (k,k,B)
                    w_mat = kern[c]                       # (k,k)
                    mask = (w_mat != 0)
                    for u in range(k):
                        for v in range(k):
                            if mask[u, v]:
                                vec = patch[u, v] * w_mat[u, v]
                                vecs.append(vec)

                # No active connection → uncertain
                if not vecs:
                    output[i, j, m] = 0
                else:
                    # 1) Sum into a B‑vector (or B+1 with carry)
                    summed = vector_ripple_add(vecs, adder, bits=B)
                    # 2) Map to single trit
                    output[i, j, m] = map_vector_to_trit(summed, threshold)

    return output

# Load one MNIST image (first example from test set)
transform = transforms.Compose([transforms.ToTensor()])
mnist = MNIST(root="./data", train=False, download=True, transform=transform)
image, label = mnist[0]
image_np = (image[0].numpy() * 255).astype(np.uint8)  # shape (28, 28)

# Encode with uncertainty using sigma (e.g., σ=2.0)
sigma = 2.0
encoded_img = encode_image_with_uncertainty(image_np, sigma)  # shape (28, 28, 8)
feature_map = encoded_img[:, :, None, :]  # Add channel dim: (H, W, C=1, B=8)

# Define a random ternary kernel: M=1 filter, C=1 channel, k=3
# Horizontal edge detector
kernels =  np.array([
    [[[0, 1, 0], [1, -1, 1], [0, 1, 0]]]
])


# Apply ternary convolution
output = ternary_convolution(feature_map, kernels, threshold=3)

# === Visualization ===
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

axs[0].imshow(image_np, cmap='gray')
axs[0].set_title(f"Original MNIST (Label: {label})")
axs[0].axis('off')

axs[1].imshow(output[:, :, 0], cmap='gray', vmin=-1, vmax=1)
axs[1].set_title("Ternary Convolution Output")
axs[1].axis('off')

plt.tight_layout()
plt.show()
# # === Example usage ===
# if __name__ == "__main__":
#     # Dummy feature map: 5×5 spatial, 3 channels, B=8 bits
#     fmap = np.random.choice([-1,0,1], size=(5,5,3,8), p=[0.2,0.6,0.2])
#     # Dummy kernels: 2 filters, 3 channels, 3×3
#     kern = np.random.choice([-1,0,1], size=(2,3,3,3), p=[0.3,0.4,0.3])
#
#     out = ternary_convolution(fmap, kern)
#     print("Output shape:", out.shape)         # → (3,3,2)
#     print("Sample output at (0,0):", out[0,0]) # two trits for 2 filters