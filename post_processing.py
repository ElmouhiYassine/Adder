import numpy as np


def interval_range_classification(conv4, theta = 1):
    """
    conv4: ndarray of shape (H, W, 4), each entry in {-1,0,1}
    theta: integer threshold in [0, 2^4)

    Returns: ndarray of shape (H, W) with values in {-1, 0, 1}
      -1 = output 0 (false),
       0 = unknown,
       1 = output 1 (true)
    """
    H, W, _ = conv4.shape
    out = np.zeros((H, W), dtype=int)

    # Precompute bit weights
    weights = np.array([1, 2, 4, 8])  # 2^0,2^1,2^2,2^3

    # Map trit → interval bounds: -1→[0,0], 0→[0,1], 1→[1,1]
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
    out[L > theta] = 1
    out[(U >= theta) & (L <= theta)] = 0

    return out
