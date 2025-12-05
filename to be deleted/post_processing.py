import numpy as np


def interval_range_classification(conv4, theta = 3):

    H, W, _ = conv4.shape
    out = np.zeros((H, W), dtype=int)

    # bit weights
    weights = np.array([1, 2, 4, 8])  # 2^0,2^1,2^2,2^3

    # Map trit → interval : -1→[0,0], 0→[0,1], 1→[1,1]
    # We'll build two arrays L and U:
    L = np.zeros((H, W), dtype=int)
    U = np.zeros((H, W), dtype=int)

    for bit in range(4):
        b = conv4[..., bit]
        w = weights[bit]
        # lower bound contribution: only 1→1
        L += (b == 1).astype(int) * w
        # upper bound contribution: both 0 and 1 map to 1
        U += (b != -1).astype(int) * w

    # Decision per pixel
    #   if U < theta  → output false (-1)
    #   if L > theta  → output true  (1)
    #   otherwise     → unknown    (0)

    out[U < theta] = -1
    out[L >= theta] = 1

    return out


def min_assumption(conv4, theta=3):

    H, W, _ = conv4.shape
    out = np.zeros((H, W), dtype=int)

    # Weights for each bit position (LSB to MSB)
    weights = np.array([1, 2, 4, 8])  # 2^0, 2^1, 2^2, 2^3

    # Convert ternary to binary representation (0 → -1)
    bin_conv = np.where(conv4 == 0, -1, conv4)  # Replace 0 with -1
    bin_conv = np.where(bin_conv == -1, 0, bin_conv)  # Now: -1→0, 1→1

    # Calculate decimal value for each pixel
    decimal_vals = np.sum(bin_conv * weights, axis=2)

    # Threshold comparison
    out[decimal_vals < theta] = -1
    out[decimal_vals >= theta] = 1

    return out


def max_assumption(conv4, theta=3):
    """
    Treats uncertain trits (0) as arithmetic 1 (true)
    Returns binary output: -1 (false) or 1 (true)
    """
    H, W, _ = conv4.shape
    out = np.zeros((H, W), dtype=int)

    # Weights for each bit position (LSB to MSB)
    weights = np.array([1, 2, 4, 8])  # 2^0, 2^1, 2^2, 2^3

    # Convert ternary to binary representation (0 → 1)
    bin_conv = np.where(conv4 == 0, 1, conv4) # Replace 0 with 1
    bin_conv = np.where(bin_conv == -1, 0, bin_conv)

    # Calculate decimal value for each pixel
    decimal_vals = np.sum(bin_conv * weights, axis=2)

    # Threshold comparison
    out[decimal_vals < theta] = -1
    out[decimal_vals >= theta] = 1

    return out
