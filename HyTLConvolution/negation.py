import itertools
import numpy as np
from Adders.sobocinski_adder import sobocinski_ripple
from Adders.strong_kleene_adder import SK_ripple_add

def twos_complement_(vec, width, adder_func):
    # -------------------------------------------------
    # 1) Pad to full width
    # -------------------------------------------------
    x = vec[:] + [-1] * (width - len(vec))  # -1 = definite 0

    # -------------------------------------------------
    # 2) Bitwise flip
    #    1 -> -1
    #   -1 ->  1
    #    0 ->  0   (uncertainty preserved)
    # -------------------------------------------------
    flipped = []
    for t in x:
        if t == 1:
            flipped.append(-1)
        elif t == -1:
            flipped.append(1)
        else:
            flipped.append(0)

    # -------------------------------------------------
    # 3) Add +1 using the uncertain adder
    # -------------------------------------------------
    one = [1] + [-1] * (width - 1)   # +1 in LSB-first ternary

    result = adder_func([one, flipped])

    # Ignore overflow beyond width
    return result[:width]

# ----------------------------
# Bounds for "uncertain-binary"
# ----------------------------
def bounds_uncertain_binary(uvec_lsb: list[int], width: int) -> tuple[int, int]:
    """
    Returns (min, max) for an uncertain-binary vector.
    -1 means definite 0
     1 means definite 1
     0 means uncertain {0,1}
    """
    vec = uvec_lsb + [-1] * (width - len(uvec_lsb))
    mn = 0
    mx = 0
    for i, t in enumerate(vec[:width]):
        if t == 1:
            mn += (1 << i)
            mx += (1 << i)
        elif t == 0:
            # could be 0 or 1
            mx += (1 << i)
    return mn, mx

def eval_negation_once(uvec_lsb: list[int], width: int) -> dict:
    x_min, x_max = bounds_uncertain_binary(uvec_lsb, width)

    neg = twos_complement_(uvec_lsb, width)
    neg_min, neg_max = bounds_uncertain_binary(neg, width)

    # Ground-truth interval after negation
    gt_min = -x_max
    gt_max = -x_min

    # How far are we from the correct interval?
    # (0 means perfect)
    err_low  = abs(neg_min - gt_min)
    err_high = abs(neg_max - gt_max)

    # Also check containment (ideal is: neg_interval == gt_interval)
    contains = (neg_min <= gt_min) and (neg_max >= gt_max)
    inside   = (neg_min >= gt_min) and (neg_max <= gt_max)

    return {
        "x": uvec_lsb,
        "x_bounds": (x_min, x_max),
        "neg": neg,
        "neg_bounds": (neg_min, neg_max),
        "gt_bounds": (gt_min, gt_max),
        "err_low": err_low,
        "err_high": err_high,
        "contains_gt": contains,  # neg interval is a superset of correct
        "inside_gt": inside,      # neg interval is a subset of correct (danger!)
    }

# ----------------------------
# Batch evaluation
# ----------------------------
def eval_negation_batch(width: int, n_bits: int = None, include_uncertain=True):
    """
    Evaluate negation over many vectors.
    - width: total width
    - n_bits: if set, vary only first n_bits and pad rest with -1
    """
    if n_bits is None:
        n_bits = width

    choices = [-1, 0, 1] if include_uncertain else [-1, 1]

    worst = None
    stats = {
        "count": 0,
        "perfect": 0,
        "avg_err_low": 0.0,
        "avg_err_high": 0.0,
        "inside_gt_count": 0,
        "contains_gt_count": 0,
    }

    for core in itertools.product(choices, repeat=n_bits):
        x = list(core) + [-1] * (width - n_bits)
        r = eval_negation_once(x, width)

        stats["count"] += 1
        stats["avg_err_low"] += r["err_low"]
        stats["avg_err_high"] += r["err_high"]

        if r["err_low"] == 0 and r["err_high"] == 0:
            stats["perfect"] += 1
        if r["inside_gt"]:
            stats["inside_gt_count"] += 1
        if r["contains_gt"]:
            stats["contains_gt_count"] += 1

        score = r["err_low"] + r["err_high"]
        if worst is None or score > (worst["err_low"] + worst["err_high"]):
            worst = r

    stats["avg_err_low"] /= max(1, stats["count"])
    stats["avg_err_high"] /= max(1, stats["count"])

    return stats, worst

# print(twos_complement_([0,1],9,sobocinski_ripple))





from typing import List, Tuple

def bounds_of_uncertain_twos_complement(trits_lsb: List[int]) -> Tuple[int, int]:
    """
    Compute exact integer bounds (min,max) of an uncertain 2's-complement word.

    Coding (LSB-first):
      trit = +1  -> bit is definitely 1
      trit = -1  -> bit is definitely 0
      trit =  0  -> bit is uncertain in {0,1}

    Numeric value uses standard two's complement weights:
      bits 0..n-2 have weight +2^i
      bit  n-1   has weight -2^(n-1)   (the MSB)
    """
    n = len(trits_lsb)
    mn = 0
    mx = 0
    for i, t in enumerate(trits_lsb):
        w = (1 << i)
        if i == n - 1:
            w = -w  # MSB is negative weight

        if t == +1:
            # definitely contributes weight
            mn += w
            mx += w
        elif t == -1:
            # definitely contributes 0
            pass
        else:
            # uncertain: contributes either 0 or w
            # if w>0: range [0, w]
            # if w<0: range [w, 0]
            mn += min(0, w)
            mx += max(0, w)

    # ensure ordered
    if mn > mx:
        mn, mx = mx, mn
    return mn, mx


def int_to_binary(n: int, width: int):
    return [+1 if (n >> b) & 1 else -1 for b in range(width)]


def negate(trits_lsb: List[int], adder) -> List[int]:
    """
    Negate an uncertain 2's-complement word by:
      1) computing bounds [xmin, xmax]
      2) negating them -> [-xmax, -xmin]
      3) encoding both endpoints in 2's complement
      4) marking differing bits as uncertain (0)

    Guarantee:
      if xmin <= x <= xmax then output represents some y with
      -xmax <= y <= -xmin (i.e., y = -x).
    """
    width = len(trits_lsb)

    xmin, xmax = bounds_of_uncertain_twos_complement(trits_lsb)

    bmax = int_to_binary(xmax, width)
    bmin = int_to_binary(xmin, width)

    lo = twos_complement_(bmax,width,adder)
    hi = twos_complement_(bmin,width,adder)

    neg = []
    for a, b in zip(lo, hi):
        neg.append(a if a == b else 0)
    return neg


# ------------------------
# quick test with your ex
# ------------------------
x = [0, 0, 0, 1, -1, -1]  # LSB-first
xmin, xmax = bounds_of_uncertain_twos_complement(x)
print("x bounds:", xmin, xmax)  # expect 8..15

nx = negate(x,SK_ripple_add)
nmin, nmax = bounds_of_uncertain_twos_complement(nx)
print("neg trits:", nx)
print("neg bounds:", nmin, nmax)  # expect -15..-8
