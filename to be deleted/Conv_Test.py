from Adders.balanced_ternary_adder import balanced_ternary_add
from Adders.pesi_op_adder import pess_op_ripple
from Adders.SK_Quasi_adder import ripple_quasi
from Adders.lukasiewicz_adder import lukasiewicz_ripple
from Adders.Ternary_New_Adder import get_Adder
import numpy as np
import math
from typing import Tuple, Callable, Any

# ---------- helpers ----------
def needed_bt_width(max_abs_sum: int) -> int:
    # smallest n such that (3**n - 1)//2 >= max_abs_sum
    n = 1
    while (3**n - 1)//2 < max_abs_sum:
        n += 1
    return n

def fit_width(vec, width, pad=0):
    """Ensure vec (LSB-first) fits exactly `width` digits (trim/pad)."""
    if len(vec) >= width:
        return vec[:width]
    return vec + [pad] * (width - len(vec))

def int_to_binary(n: int, width: int):
    """Encode integer n as {+1,-1} (LSB-first) for uncertain-binary adder."""
    out = []
    for i in range(width):
        out.append(+1 if (n & (1 << i)) else -1)
    return out

def int_to_balanced_ternary(n: int, min_len: int = None) -> list[Any] | list[int]:
    """
    Convert integer n to *arbitrary-length* balanced-ternary (LSB-first).
    Digits in {-1, 0, +1}. If min_len given, pad with zeros to that length.
    """
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
            elif r == 2:  # treat as -1 with carry
                vec.append(-1)
                num = (num + 1) // 3
    if min_len is not None and len(vec) < min_len:
        vec += [0] * (min_len - len(vec))
    return vec

def flip_only_extend_LSBfirst(u_lsb: list, K1: int):
    """
    One's-complement flip for uncertain-binary trits: +1<->-1, 0 stays 0.
    Input:  K1-digit vector LSB-first.
    Output: (K1+1)-digit vector LSB-first (prepends a -1 baseline on MSB side, then flips all).
    """
    msb = list(reversed(u_lsb))
    msb = [-1] + msb  # extend by a leading -1 baseline (MSB side)
    flipped_msb = []
    for b in msb:
        if b == +1:
            flipped_msb.append(-1)
        elif b == -1:
            flipped_msb.append(+1)
        else:
            flipped_msb.append(0)
    return list(reversed(flipped_msb))  # back to LSB-first, length K1+1

# small helpers for U certain-path math
def bits_to_int(lsb_vec: list[int]) -> int:
    """Map {-1,+1}→{0,1} and build integer from LSB-first vector."""
    val = 0
    for b, t in enumerate(lsb_vec):
        if t == +1:
            val |= (1 << b)
    return val

def int_to_bits(n: int, width: int) -> list[int]:
    """Integer → {-1,+1} bits (LSB-first), width-limited."""
    return [+1 if (n >> b) & 1 else -1 for b in range(width)]

def U_pass(U: np.ndarray,
           kernel: np.ndarray,
           ripple_add: Callable,
           U_EXT_WIDTH: int = 10,
           carry_policy: str = "definite") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fixed-width uncertainty pass over U with a hybrid adder:
      - If all trits in the window are certain (∈{-1,+1}), use exact *signed* integer addition.
      - Otherwise, use SK ripple with flip + (+1 per negative tap) correction.

    Carry comes from bits [K1 .. U_EXT_WIDTH-2] (exclude final sign bit).
    If uncertain bits exist in the carry window, resolve via `carry_policy`:
      - 'definite' : mark as uncertain (mask=0) and set carry=0
      - 'max'      : replace 0s by +1 (optimistic bound), mask=0
      - 'min'      : replace 0s by -1 (pessimistic bound → contributes 0), mask=0
      - 'center'   : 0 → 0.5 (expected value), sum and round to nearest int, mask=0

    Returns:
      U_trunc   : (H', W', K1)       -- low K1 trits (LSB-first)
      carry_vals: (H', W')           -- integer carry (absolute units of 1)
      carry_mask: (H', W')           -- +1 definite, 0 uncertain, -1 definite zero
    """
    import math

    H, W, K1 = U.shape
    k = kernel.shape[0]
    out_H, out_W = H - k + 1, W - k + 1

    U_trunc    = np.full((out_H, out_W, K1), -1, dtype=int)
    carry_vals = np.zeros((out_H, out_W), dtype=int)
    carry_mask = np.full((out_H, out_W), -1, dtype=int)

    # extend the vector, adding -1
    def pad_to_ext(vec_lsb: list[int]) -> list[int]:
        if len(vec_lsb) < U_EXT_WIDTH:
            return vec_lsb + [-1] * (U_EXT_WIDTH - len(vec_lsb))
        return vec_lsb[:U_EXT_WIDTH]

    # flip -1 to 1, 1 to -1
    def flip_only_pad(vec_lsb: list[int]) -> list[int]:
        vec_lsb = pad_to_ext(vec_lsb)
        flipped = [(-1 if t == +1 else (+1 if t == -1 else 0)) for t in vec_lsb]
        return flipped

    def bits_to_int(lsb_vec: list[int]) -> int:
        """Map {-1,+1}→{0,1} and build integer from LSB-first vector."""
        val = 0
        for b, t in enumerate(lsb_vec):
            if t == +1:
                val |= (1 << b)
        return val

    def int_to_bits(n: int, width: int) -> list[int]:
        """Integer → {-1,+1} bits (LSB-first), width-limited."""
        return [+1 if (n >> b) & 1 else -1 for b in range(width)]

    def twos_complement_(uvec_lsb: list[int], U_EXT_WIDTH: int) -> list[int]:

        # 1. Pad the input vector and flip LSB/MSB order for easier traversal
        full_vector = uvec_lsb + [-1] * (U_EXT_WIDTH - len(uvec_lsb))

        # 2. Iterate and apply the rule
        result = []
        found_one = False

        for i in range(len(full_vector)):
            trit = full_vector[i]

            if not found_one:
                # Step 1 & 2: Copy until the first +1
                if trit == +1:
                    result.append(+1)
                    found_one = True
                elif trit == -1:
                    result.append(-1)
                elif trit == 0:
                    result.append(0)
            else:
                # Step 3: Invert all remaining bits
                if trit == +1:
                    result.append(-1)
                elif trit == -1:
                    result.append(+1)
                elif trit == 0:
                    # Uncertainty propagates through inversion
                    result.append(0)

        return result

    def resolve_carry(c_segment: list[int], start_bit: int) -> tuple[int, int]:
        """
        Resolve carry from c_segment according to carry_policy.
        Returns (carry_val:int, carry_mask:int)
        """
        # --- Setup for all uncertain policies ---
        # We need the index 'i' to find the sign bit
        num_bits = len(c_segment)
        sign_bit_index = num_bits - 1  # The index of the sign bit
        # No uncertainty: definite
        has_zero = any(t == 0 for t in c_segment)
        # sign = c_segment[-1] # This line is correct, -1 is the last element

        # No uncertainty: definite
        if not has_zero:
            val = 0

            # We must iterate by index to know which bit is the sign bit
            for i, trit in enumerate(c_segment):

                # bit_pos is the actual power of 2 (e.g., K1, K1+1, ...)
                bit_pos = start_bit + i

                if trit == +1:  # This means the binary bit is 1
                    if i == sign_bit_index:
                        # This is the sign bit, and it's 1, so SUBTRACT its value
                        val -= (1 << bit_pos)
                    else:
                        # This is a magnitude bit, and it's 1, so ADD its value
                        val += (1 << bit_pos)

                # If trit == -1 (binary bit is 0), we add/subtract 0,
                # so we do nothing, which is correct.

            return val, 1

        # Uncertain: apply policy
        if carry_policy == "definite":
            return 0, 1

        elif carry_policy == "max":
            # Maximize signed value:
            # Mag bits 0 -> +1 (add), Sign bit 0 -> -1 (make positive)
            val = 0
            for i, trit in enumerate(c_segment):
                bit_pos = start_bit + i
                is_sign_bit = (i == sign_bit_index)

                effective_trit = trit
                if trit == 0:
                    effective_trit = -1 if is_sign_bit else +1  # -1 for sign bit, +1 for mag

                if effective_trit == +1:
                    if is_sign_bit:
                        val -= (1 << bit_pos)
                    else:
                        val += (1 << bit_pos)
            return val, 1

        elif carry_policy == "min":
            # Minimize signed value:
            # Mag bits 0 -> -1 (add 0), Sign bit 0 -> +1 (make negative)
            val = 0
            for i, trit in enumerate(c_segment):
                bit_pos = start_bit + i
                is_sign_bit = (i == sign_bit_index)

                effective_trit = trit
                if trit == 0:
                    effective_trit = +1 if is_sign_bit else -1  # +1 for sign bit, -1 for mag

                if effective_trit == +1:
                    if is_sign_bit:
                        val -= (1 << bit_pos)
                    else:
                        val += (1 << bit_pos)
            return val, 1

        elif carry_policy == "center":
            # Expected value: -1->0, +1->1, 0->0.5
            # Apply correct weights (subtract for sign bit)
            val_float = 0.0
            for i, trit in enumerate(c_segment):
                bit_pos = start_bit + i
                is_sign_bit = (i == sign_bit_index)

                p1 = 1.0 if trit == +1 else (0.0 if trit == -1 else 0.5)

                if is_sign_bit:
                    val_float -= p1 * (1 << bit_pos)
                else:
                    val_float += p1 * (1 << bit_pos)

            val = int(round(val_float))
            return val, 1

        else:
            # Fallback to 'definite'
            return 0, 1

    for i in range(out_H):
        for j in range(out_W):
            # Collect K1-bit U snippets for the k×k patch
            taps = []
            all_certain = True
            for dx in range(k):
                for dy in range(k):
                    kval = kernel[dx, dy]
                    if kval == 0:
                        continue
                    uvec = U[i + dx, j + dy, :].tolist()  # length K1
                    if any(t == 0 for t in uvec):
                        all_certain = False
                    taps.append((kval, uvec))

            if not taps:
                continue

            if all_certain:
                # === Exact signed integer path (no uncertainty) ===
                base = (1 << K1)
                acc_int_sum = 0  # signed sum over low K1 bits
                for kval, uvec in taps:
                    term = bits_to_int(uvec)  # 0..(2^K1-1)
                    acc_int_sum += (term if kval == +1 else -term)

                # Split: acc_int_sum = low + carry_units*base, 0 <= low < base
                carry_units = math.floor(acc_int_sum / base)
                low = acc_int_sum - carry_units * base

                # Emit low K1
                U_trunc[i, j, :] = np.array(int_to_bits(low, K1), dtype=int)

                # Signed carry, expressed in absolute units
                carry_vals[i, j] = carry_units * base
                carry_mask[i, j] = (+1 if carry_units != 0 else -1)

            else:
                # === Uncertain path: SK ripple with correction ===
                terms = []
                neg_count = 0
                for kval, uvec in taps:
                    if kval == +1:
                        term = pad_to_ext(uvec)
                    else:
                        term = twos_complement_(uvec, U_EXT_WIDTH)
                        neg_count += 1
                    terms.append(term)

                acc = terms[0]
                for t in terms[1:]:
                    acc = ripple_add([acc, t])
                    acc = pad_to_ext(acc)

                # if neg_count:
                #     corr = int_to_binary(neg_count, width=U_EXT_WIDTH)
                #     acc  = ripple_add([acc, corr])
                #     acc  = pad_to_ext(acc)

                # Low K1 out
                U_trunc[i, j, :] = acc[:K1]
                # print(U_trunc[i, j, :])

                # Carry window [K1 .. U_EXT_WIDTH-2]
                c_segment = acc[K1:U_EXT_WIDTH]
                val, mask = resolve_carry(c_segment, start_bit=K1)
                carry_vals[i, j] = val
                carry_mask[i, j] = mask

    return U_trunc, carry_vals, carry_mask


# ---------- V pass (balanced ternary) ----------
def V_pass(V: np.ndarray,
           kernel: np.ndarray,
           balanced_ternary_add: Callable) -> np.ndarray:
    """
    Balanced-ternary convolution on fixed-width K2 vectors (LSB-first).
    V: (H, W, K2) with digits in {-1,0,+1}, LSB-first across K2.
    kernel: (k, k) in {-1,0,+1}.
    Returns: (H-k+1, W-k+1, K2)
    """
    H, W, K2 = V.shape
    k = kernel.shape[0]
    out_H, out_W = H - k + 1, W - k + 1
    V_out = np.zeros((out_H, out_W, K2), dtype=int)

    for i in range(out_H):
        for j in range(out_W):
            acc = None
            for dx in range(k):
                for dy in range(k):
                    kval = kernel[dx, dy]
                    if kval == 0:
                        continue
                    vec = V[i + dx, j + dy, :].tolist()  # LSB-first
                    if kval == -1:
                        vec = [-t for t in vec]  # BT negation

                    if acc is None:
                        acc = vec
                    else:
                        # give +1 digit headroom during accumulation
                        L = max(len(acc), len(vec)) + 1
                        acc = balanced_ternary_add(fit_width(acc, L, 0),
                                                   fit_width(vec, L, 0))
            if acc is not None:
                V_out[i, j, :] = np.array(fit_width(acc, K2, pad=0), dtype=int)

    return V_out

# ---------- Full convolution ----------
def convolution(X: np.ndarray,
                kernel: np.ndarray,
                ripple_add: Callable,
                balanced_ternary_add: Callable,
                K1: int = 5,
                K2: int = 5,
                return_info: bool = False):
    """
    Full (K1+K2)-trit convolution:
      1) U-pass: low K1 trits + signed carry (in units of 2^K1).
      2) V-pass: balanced-ternary accumulation.
      3) Inject *signed* carry into V.
      4) Recombine to (H-k+1, W-k+1, K1+K2).
    """
    H, W, _ = X.shape
    U = X[:, :, :K1]      # (H, W, K1), LSB-first
    V = X[:, :, K1:]      # (H, W, K2), LSB-first (BT digits)

    U_trunc, carry_vals, carry_mask = U_pass(U, kernel, ripple_add, carry_policy = "center")
    V_out = V_pass(V, kernel, balanced_ternary_add)
    # print(carry_vals)
    H_out, W_out, _ = V_out.shape
    V_final = V_out.copy()

    for i in range(H_out):
        for j in range(W_out):
            amt = int(carry_vals[i, j])  # signed
            if amt != 0 and carry_mask[i, j] != 0:  # definite, nonzero
                off_vec = int_to_balanced_ternary(abs(amt), K2)
                if amt < 0:
                    off_vec = [-d for d in off_vec]  # negate in BT
                curr = fit_width(V_final[i, j, :].tolist(), K2, 0)
                tmp  = balanced_ternary_add(curr, off_vec)
                V_final[i, j, :] = np.array(fit_width(tmp, K2, pad=0), dtype=int)

    Y = np.concatenate([U_trunc, V_final], axis=2)
    if return_info:
        return Y, {"U_trunc": U_trunc, "carry_vals": carry_vals, "carry_mask": carry_mask, "V_pre": V_out}
    return Y

# ===========================================================
# ===============  TEST + HELPERS (below)  ==================
# ===========================================================

# ---- decoding helpers for comparison ----
def phi_center(t):
    """Center map for uncertain binary trit: -1→0, 0→0.5, +1→1."""
    return 0.0 if t == -1 else (0.5 if t == 0 else 1.0)

def decode_U_center(u_lsb: np.ndarray) -> float:
    """Sum center value of U low bits as a real number with base-2 weights."""
    val = 0.0
    for i, d in enumerate(u_lsb.tolist()):
        val += phi_center(d) * (2 ** i)
    return val

def decode_V_integer(v_bt_lsb: np.ndarray) -> int:
    """Evaluate balanced-ternary digits to an integer."""
    total = 0
    for i, d in enumerate(v_bt_lsb.tolist()):
        total += int(d) * (3 ** i)
    return total

def decode_Y_numeric_center(Y: np.ndarray, K1: int, K2: int) -> np.ndarray:
    """
    Convert framework output Y (H',W',K1+K2) to a single numeric map:
    number = center(U_low) + integer(V_bt).
    """
    H, W, _ = Y.shape
    out = np.zeros((H, W), dtype=float)
    for i in range(H):
        for j in range(W):
            u = Y[i, j, :K1]
            v = Y[i, j, K1:K1+K2]
            out[i, j] = decode_U_center(u) + decode_V_integer(v)
    return out

# ---- classic 2D valid convolution on ints ----
def conv2d_valid_int(img: np.ndarray, ker: np.ndarray) -> np.ndarray:
    """Valid conv (no padding) on integer image."""
    H, W = img.shape
    k = ker.shape[0]
    out = np.zeros((H - k + 1, W - k + 1), dtype=int)
    for i in range(H - k + 1):
        for j in range(W - k + 1):
            block = img[i:i+k, j:j+k]
            out[i, j] = int(np.sum(block * ker))
    return out

# ---- uncertainty mask policies ----
def uncertainty_mask_threshold(noise_abs: int, K1: int):
    """
    Mark bits b with 2^b <= |noise| as uncertain.
    e.g., |noise|=5 → bits {0,1,2} uncertain for K1>=3.
    """
    mask = np.ones(K1, dtype=bool)  # True=certain; will set uncertain to False
    b = 0
    while b < K1 and (2 ** b) <= noise_abs:
        mask[b] = False
        b += 1
    return mask  # bool: True=certain, False=uncertain

def uncertainty_mask_even(noise_abs: int, K1: int):
    """
    Mark even bits up to floor(log2(|noise|)) as uncertain.
    """
    mask = np.ones(K1, dtype=bool)
    if noise_abs <= 0:
        return mask
    msb = int(math.floor(math.log2(noise_abs)))
    for b in range(0, min(K1, msb + 1), 2):
        mask[b] = False
    return mask

def uncertainty_mask_bitset(noise_abs: int, K1: int):
    """
    Mark bits whose positions are set in |noise|’s binary expansion.
    """
    mask = np.ones(K1, dtype=bool)
    for b in range(K1):
        if (noise_abs >> b) & 1:
            mask[b] = False
    return mask

# ---- encode image (uint8) to (U,V) tensors ----
def encode_uint8_to_UV(img_noisy: np.ndarray,
                       noise_abs_map: np.ndarray,
                       K1: int,
                       K2: int,
                       uncertainty_policy: str = "threshold"):
    """
    Encode a noisy uint8 image into U (uncertain-binary, K1 LSBs) and V (BT, K2 digits).
    - For each pixel p:
      * U low bits: if a bit is 'certain' by mask → +1 for 1, -1 for 0; if 'uncertain' → 0.
      * V digits: encode the upper value as an integer in balanced ternary (length K2).
        We use v_int = p >> K1 (upper part) by default.
    """
    H, W = img_noisy.shape
    U = np.zeros((H, W, K1), dtype=int)
    V = np.zeros((H, W, K2), dtype=int)

    # choose mask function
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
            mask = mask_fn(nz, K1)  # True=certain, False=uncertain

            # U: K1 low bits (LSB-first)
            for b in range(K1):
                if not mask[b]:
                    U[i, j, b] = 0
                else:
                    U[i, j, b] = (+1 if ((p >> b) & 1) else -1)

            # V: upper part as integer in BT (LSB-first)
            upper_abs = (p >> K1) << K1
            V[i, j, :] = fit_width(int_to_balanced_ternary(upper_abs), K2, pad=0)

    # Stack to X (H,W,K1+K2), LSB-first within blocks
    X = np.concatenate([U, V], axis=2)
    return X, U, V

# ---- MNIST loader (with fallback) ----
def load_xmnist_first_N(dataset: str = "mnist", n_samples: int = 64) -> np.ndarray:
    """
    Load first N images from MNIST or Fashion-MNIST as uint8 arrays (28x28).
    dataset ∈ {"mnist", "fashion"}. Falls back to random if torchvision missing.
    """
    try:
        from torchvision import transforms
        if dataset.lower() == "mnist":
            from torchvision.datasets import MNIST as DS
        elif dataset.lower() == "fashion":
            from torchvision.datasets import FashionMNIST as DS
        else:
            raise ValueError("dataset must be 'mnist' or 'fashion'")

        ds = DS(root="./data", train=True, download=True, transform=transforms.ToTensor())
        imgs = []
        for k in range(min(n_samples, len(ds))):
            x, _ = ds[k]                      # x: torch.Tensor [1,28,28] in [0,1]
            arr = (x.numpy()[0] * 255.0).round().astype(np.uint8)
            imgs.append(arr)
        return np.stack(imgs, axis=0)
    except Exception:
        rng = np.random.default_rng(0)
        return rng.integers(low=0, high=256, size=(n_samples, 28, 28), dtype=np.uint8)

# ---- experiment runner ----
def psnr(mse, max_val=255.0):
    return float("inf") if mse == 0 else 10.0 * math.log10((max_val ** 2) / mse)

def run_mnist_noise_experiment(n_images: int = 32,
                               K1: int = 4,
                               K2: int = 4,
                               kernel: np.ndarray = None,
                               noise_sigma: float = 5.0,
                               noise_dist: str = "gaussian",
                               uncertainty_policy: str = "threshold",
                               dataset: str = "fashion",
                               seed: int = 1234):
    """
    Compare 3 pipelines on MNIST/Fashion:
      (A) Clean baseline conv on original image.
      (B) Naïve conv on noisy int image (no uncertainty handling).
      (C) Framework conv on (U,V) encoded noisy image + decode to numeric center.

    Report MAE/MSE/PSNR of (B) and (C) vs (A).
    """
    rng = np.random.default_rng(seed)
    imgs = load_xmnist_first_N(dataset=dataset, n_samples=n_images)

    # kernel (3x3) — default: all-ones
    if kernel is None:
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    k = kernel.shape[0]
    assert kernel.shape == (k, k)

    mae_naive, mse_naive, mae_fw, mse_fw = [], [], [], []

    for idx in range(imgs.shape[0]):
        img = imgs[idx].astype(int)  # 0..255

        # --- noise injection ---
        if noise_dist == "gaussian":
            noise = rng.normal(loc=0.0, scale=noise_sigma, size=img.shape)
        elif noise_dist == "uniform":
            noise = rng.uniform(low=-noise_sigma, high=noise_sigma, size=img.shape)
        else:
            raise ValueError("noise_dist must be 'gaussian' or 'uniform'")

        noise = noise.round().astype(int)
        noise_abs = np.abs(noise)

        img_noisy = np.clip(img + noise, 0, 255).astype(int)

        # sanity when sigma=0
        if noise_sigma == 0.0:
            assert np.all(noise_abs == 0), "sigma=0 should produce zero noise_abs"

        # --- Baseline clean conv ---
        base = conv2d_valid_int(img, kernel)  # int

        # --- Naïve noisy conv (no uncertainty) ---
        naive = conv2d_valid_int(img_noisy, kernel)  # int

        # --- Our framework ---
        X, U, V = encode_uint8_to_UV(img_noisy, noise_abs, K1, K2,
                                     uncertainty_policy=uncertainty_policy)

        Y = convolution(X, kernel, ripple_quasi, balanced_ternary_add, K1=K1, K2=K2)
        fw_map = decode_Y_numeric_center(Y, K1, K2)  # float

        # metrics vs clean baseline (same spatial size)
        mae_naive.append(np.mean(np.abs(naive - base)))
        mse_naive.append(np.mean((naive - base) ** 2))
        mae_fw.append(np.mean(np.abs(fw_map - base)))
        mse_fw.append(np.mean((fw_map - base) ** 2))

    mae_naive = float(np.mean(mae_naive))
    mse_naive = float(np.mean(mse_naive))
    mae_fw = float(np.mean(mae_fw))
    mse_fw = float(np.mean(mse_fw))

    print("=== MNIST noise experiment ===")
    print(f"Images               : {n_images}")
    print(f"K1/K2                : {K1}/{K2}")
    print(f"Kernel               :\n{kernel}")
    print(f"Noise dist / sigma   : {noise_dist} / {noise_sigma}")
    print(f"Uncertainty policy   : {uncertainty_policy}")
    print("---- Results vs CLEAN baseline ----")
    print(f"Naïve (no handling):  MAE={mae_naive:.3f}  MSE={mse_naive:.3f}  PSNR={psnr(mse_naive):.2f} dB")
    print(f"Framework (U,V):     MAE={mae_fw:.3f}  MSE={mse_fw:.3f}  PSNR={psnr(mse_fw):.2f} dB")
    print("(Higher PSNR / lower MAE/MSE is better)")


    return {
        "noise_sigma": float(noise_sigma),
        "mae_naive": mae_naive,
        "mse_naive": mse_naive,
        "mae_fw": mae_fw,
        "mse_fw": mse_fw
    }
# ===========================================================
# ======================  HOW TO RUN  =======================
# ===========================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    kernel = np.array([[0, -1, 1],
                       [-1, -1, 1],
                       [0, 0, -1]])

    K2 = 8
    sum_abs_kernel = int(np.sum(np.abs(kernel)))
    max_abs_sum = sum_abs_kernel * 255
    K2_needed = needed_bt_width(max_abs_sum)
    K2 = max(K2, K2_needed)

    noise_levels = [0,5,10,15,20,25,30,31]

    mae_naive_list = []
    mae_fw_list = []

    for sigma in noise_levels:
        res = run_mnist_noise_experiment(
            n_images=100,
            kernel=kernel,
            K1=5, K2=10,
            noise_sigma=sigma,
            noise_dist="gaussian",
            uncertainty_policy="threshold",
            dataset="mnist",
            seed=1234  # fixed seed => same noise pattern shape, scaled by σ
        )
        mae_naive_list.append(res["mae_naive"])
        mae_fw_list.append(res["mae_fw"])

    # Plot (single plot, no explicit colors)
    plt.figure()
    plt.plot(noise_levels, mae_naive_list, marker='o', label='Naïve (no handling)')
    plt.plot(noise_levels, mae_fw_list, marker='s', label='Framework (U,V)')
    plt.xlabel('Noise σ')
    plt.ylabel('MAE vs clean baseline')
    plt.title('Noise sweep on MNIST')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # NUM_KERNELS = 20
    # KERNEL_SIZE = 3
    # # Note: K1 and K2 are fixed for the inner experiment loop (K1=5, K2=10)
    #
    # noise_levels = [0, 5, 10, 15, 20, 25, 30, 31]
    #
    # # --- Data Structures for Accumulation ---
    # # Will store NUM_KERNELS lists of MAE values for each sigma level.
    # all_mae_naive_results = {sigma: [] for sigma in noise_levels}
    # all_mae_fw_results = {sigma: [] for sigma in noise_levels}
    #
    # # Store specific instances where Naive wins
    # naive_wins_data = []
    #
    # # --- Outer Loop: Generate and Test Kernels ---
    # rng = np.random.default_rng(1234)  # Use a specific RNG for reproducible kernel generation
    #
    # print(f"--- Running {NUM_KERNELS} Random Kernels across {len(noise_levels)} Noise Levels ---")
    #
    # for k_idx in range(NUM_KERNELS):
    #     # Generate a random 3x3 kernel with values in {-1, 0, 1}
    #     # rng.integers(low=-1, high=2) gives results in {-1, 0, 1}
    #     kernel = rng.integers(low=-1, high=2, size=(KERNEL_SIZE, KERNEL_SIZE))
    #
    #     # Recalculate K2 based on the new kernel (safety check, though K2=10 is fixed later)
    #     sum_abs_kernel = int(np.sum(np.abs(kernel)))
    #     max_abs_sum = sum_abs_kernel * 255
    #     # K2_needed = needed_bt_width(max_abs_sum) # Keep this commented if K2 is fixed
    #     # K2 = max(10, K2_needed)
    #
    #     print(f"\nProcessing Kernel {k_idx + 1}/{NUM_KERNELS}:\n{kernel}")
    #
    #     # --- Inner Loop: Noise Sweep ---
    #     for sigma in noise_levels:
    #         res = run_mnist_noise_experiment(
    #             n_images=100,
    #             kernel=kernel,
    #             K1=5, K2=10,
    #             noise_sigma=sigma,
    #             noise_dist="gaussian",
    #             uncertainty_policy="threshold",
    #             dataset="mnist",
    #             seed=1234  # Ensures the MNIST images and noise pattern are consistent
    #         )
    #
    #         mae_naive = res["mae_naive"]
    #         mae_fw = res["mae_fw"]
    #
    #         all_mae_naive_results[sigma].append(mae_naive)
    #         all_mae_fw_results[sigma].append(mae_fw)
    #
    #         # --- Check for Naive Win ---
    #         if mae_naive < mae_fw:
    #             naive_wins_data.append({
    #                 "kernel_index": k_idx,
    #                 "kernel": kernel.tolist(),
    #                 "sigma": sigma,
    #                 "mae_naive": mae_naive,
    #                 "mae_fw": mae_fw,
    #                 "difference": mae_fw - mae_naive
    #             })
    #
    # # --- Calculate Mean Results for Plotting ---
    # mean_mae_naive = [np.mean(all_mae_naive_results[s]) for s in noise_levels]
    # mean_mae_fw = [np.mean(all_mae_fw_results[s]) for s in noise_levels]
    #
    # # --- Plotting Mean Results ---
    # plt.figure(figsize=(10, 6))
    # plt.plot(noise_levels, mean_mae_naive, marker='o', label='Mean Naïve MAE')
    # plt.plot(noise_levels, mean_mae_fw, marker='s', label='Mean Framework (U,V) MAE')
    # plt.xlabel('Noise σ')
    # plt.ylabel(f'Mean MAE vs Clean Baseline (N={NUM_KERNELS} Kernels)')
    # plt.title('Mean Error Sweep: Uncertainty Framework vs. Naïve Baseline')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # # --- Analysis Output ---
    # print("\n" + "=" * 50)
    # print("--- ANALYSIS OF NAIVE WINS ---")
    # print(f"Total Naïve Wins Found: {len(naive_wins_data)}")
    #
    # if naive_wins_data:
    #     print("\nPossible Case(s) where Naïve MAE was better than Framework MAE:")
    #
    #     # Identify the kernel that won the most times
    #     win_counts = {}
    #     for item in naive_wins_data:
    #         k_tuple = tuple(map(tuple, item['kernel']))
    #         win_counts[k_tuple] = win_counts.get(k_tuple, 0) + 1
    #
    #     most_winning_kernel = max(win_counts, key=win_counts.get)
    #
    #     print("\n*** The Most 'Winning' Kernel Pattern: ***")
    #     print(np.array(most_winning_kernel))
    #     print(f"Wins Count: {win_counts[most_winning_kernel]} across all sigma levels.")
    #
    #     # Display the instance with the largest difference
    #     worst_fw_case = max(naive_wins_data, key=lambda x: x['difference'])
    #     print("\n*** Worst Case for Framework (Largest Win Margin for Naïve): ***")
    #     print(f"Kernel Index: {worst_fw_case['kernel_index']} (Sigma={worst_fw_case['sigma']})")
    #     print(f"Kernel:\n{np.array(worst_fw_case['kernel'])}")
    #     print(f"MAE Naïve: {worst_fw_case['mae_naive']:.4f}")
    #     print(f"MAE Framework: {worst_fw_case['mae_fw']:.4f}")
    #     print(f"Difference (Framework - Naïve): -{worst_fw_case['difference']:.4f}")
    #
    # print("=" * 50)
