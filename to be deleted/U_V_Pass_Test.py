import numpy as np
from typing import Tuple, List
from Adders.pesi_op_adder import full_op_adder
from Adders.strong_kleene_adder import strong_kleene_full_adder
from Adders.balanced_ternary_adder import balanced_ternary_add


def adder_U(a: int, b: int, cin: int) -> Tuple[int,int]:
    s,c = full_op_adder(a, b, cin)
    return s,c


def adder_V(a: int, b: int, cin: int) -> Tuple[int,int]:
    return balanced_ternary_add(a,b,c)


# ───── Balanced ternary encoding ─────
def balanced_ternary_encode(val: int, width: int = 4) -> List[int]:
    """
    Encode integer val into balanced ternary vector of length 'width'.
    Uses weights 3^(width-1), ..., 3^0.
    """
    weights = [3**i for i in reversed(range(width))]
    vec = []
    for w in weights:
        if val >= w:
            vec.append(1)
            val -= w
        elif val <= -w:
            vec.append(-1)
            val += w
        else:
            vec.append(0)
    return vec

# ───── 1. Noise & encoding ─────
def inject_uncertainty_mask(
    image: np.ndarray, sigma: float
) -> np.ndarray:
    """
    For each pixel, sample Gaussian noise and
    produce a mask of shape (9,9,4) where mask[x,y,i]=True
    iff the i-th LSB is uncertain.
    """
    H, W = image.shape
    noise = np.random.normal(0.0, sigma, size=(H, W))
    absn = np.abs(noise)
    thresholds = np.array([2**i for i in range(4)])  # [1,2,4,8]
    return absn[:, :, None] >= thresholds[None, None, :]


def encode_image(
    image: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    Given image (9×9) and mask (9×9×4), return encoded
    tensor of shape (9,9,8) with values in {-1,0,1}.
    First 4 positions = U, next 4 = V.

    U: apply uncertainty to LSBs.
    V: take 4 MSB bits as binary number, then balanced-tertiary encode.
    """
    H, W = image.shape
    encoded = np.zeros((H, W, 8), dtype=int)
    for x in range(H):
        for y in range(W):
            p = int(image[x, y])
            # get 8-bit binary (MSB first)
            bits = [(p >> i) & 1 for i in reversed(range(8))]
            # U channels (4 LSBs): bits positions 4..7
            for i in range(4):
                bit_idx = 7 - i  # LSB=bit7
                if mask[x, y, i]:
                    encoded[x, y, i] = 0
                else:
                    encoded[x, y, i] = 1 if bits[bit_idx] else -1
            # V channels: take MSB 4 bits as binary value
            msb_val = sum(bits[j] << (3 - j) for j in range(4))
            vt = balanced_ternary_encode(msb_val, width=4)
            encoded[x, y, 4:8] = vt
    return encoded

# Logical negation function with optional adder(+1)
def negate_ternary_vector(vec: List[int]) -> List[int]:
    vec.extend([-1]*4)
    flipped = [-t if t != 0 else 0 for t in vec]  # Step 1: flip the vector
    result = []
    carry = 1  # Step 2: add 1 using ternary adder

    for trit in flipped:  # Add from least significant to most
        s, carry = strong_kleene_full_adder(trit,carry , -1)
        result.append(s)

    # If carry still exists, we may need to prepend it
    if carry != -1:
        result.insert(0, carry)
    # result.insert(0, -1)

    return result

# ───── 2. Convolution utility ─────
def convolve_2stage(
    enc: np.ndarray,
    kernel: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-stage convolution on enc shape (9,9,8).
    kernel shape = (3,3).
    Returns: U_out (7,7,4), C_out (7,7,4), V_out (7,7,4).
    """
    H, W, _ = enc.shape
    out_h, out_w = H - 3 + 1, W - 3 + 1
    U_out = np.zeros((out_h, out_w, 4), dtype=int)
    C_out = np.zeros((out_h, out_w, 4), dtype=int)
    V_out = np.zeros((out_h, out_w, 4), dtype=int)

    # U-pass
    for i in range(out_h):
        for j in range(out_w):
            for b in range(4):
                trits, carry = [], -1
                for u in range(3):
                    for v in range(3):
                        a = enc[i+u, j+v, b]
                        w = kernel[u, v]
                        if w != 0:
                            trits.append(a * w)
                if trits:
                    acc = trits[0]
                    for t in trits[1:]:
                        acc, carry = adder_U(acc, t, carry)
                    U_out[i,j,b], C_out[i,j,b] = acc, carry
                else:
                    U_out[i,j,b], C_out[i,j,b] = 0, -1
    # V pass
    for i in range(out_h):
        for j in range(out_w):
            # First compute V-pass result including its own internal carry
            V_vec = []
            carry = 0  # fresh neutral carry for each (i,j)
            for b in range(4):
                trits = []
                for u in range(3):
                    for v in range(3):
                        a = enc[i + u, j + v, 4 + b]
                        w = kernel[u, v]
                        if w != 0:
                            trits.append(a * w)
                acc = trits[0]
                for t in trits[1:]:
                    acc, carry = adder_V(acc, t, carry)
                    V_vec.append(acc)
                if carry in (-1, 1):
                    V_vec.append(carry)
            # Now apply carry from U-pass if any
            for b in range(4):
                if C_out[i, j, b] == 1:
                    val = 2 ** (3 - b)
                    carry_vec = balanced_ternary_encode(val, width=4)
                    carry = 0
                    for k in range(4):
                        acc, carry = adder_V(V_vec[k], carry_vec[k], carry)
                        V_vec[k] = acc
            # Store final V_out vector
            for b in range(4):
                V_out[i, j, b] = V_vec[b]
    return U_out, V_out
# ───── 3. Reconstruction & error ─────
def reconstruct_pixel(U_trits: List[int], V_trits: List[int]) -> List[int]:
    return U_trits + V_trits


def evaluate_patch(
    I: np.ndarray, kernel: np.ndarray, sigma: float
) -> Tuple[float,float,float]:
    mask = inject_uncertainty_mask(I, sigma)
    enc  = encode_image(I, mask)
    U_out, C_out, V_out = convolve_2stage(enc, kernel)
    H, W = U_out.shape[:2]
    # reference conv
    ref = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            ref[i,j] = np.sum(I[i:i+3,j:j+3] * kernel)
    errs = []
    for i in range(H):
        for j in range(W):
            trib = list(U_out[i,j]) + list(V_out[i,j])
            errs.append(abs(reconstruct_pixel(trib) - ref[i,j]))
    errs = np.array(errs)
    return errs.min(), errs.max(), errs.mean()

if __name__ == "__main__":
    # I = np.arange(81).reshape(9,9).astype(float) * 3
    # K = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    # for sigma in [0, 1.0, 2.0, 4.0]:
    #     mn, mx, avg = evaluate_patch(I, K, sigma)
    #     print(f"σ={sigma:.1f} → err: min={mn:.1f}, max={mx:.1f}, avg={avg:.1f}")
    print(negate_ternary_vector([1,1,1,1]))  # Output: [0, 0, 1, -1]
