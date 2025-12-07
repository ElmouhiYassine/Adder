import itertools
import json
import numpy as np

# ---- your adders (replace imports if needed) ----
from Adders.pesi_op_adder import full_op_adder
from Adders.sobocinski_adder import map_quasi_adder
from Adders.strong_kleene_adder import strong_kleene_full_adder
from Adders.Ternary_New_Adder import get_Adder

adder1 = get_Adder(0)
adder_collab = get_Adder(2)
adder_tr = get_Adder(4)
adder_2_tr = get_Adder(5)

adders = {
    "Strong-Kleene": strong_kleene_full_adder,
    "Sobocinski": map_quasi_adder,
    "Bochvar external": adder1,
    "Sette": adder_collab,
    "Lukasiewicz": adder_tr,
    "Gaines-Rescher": adder_2_tr,
}
# -------------------------------------------------

# N_BITS = number of *value* bits
# REP_WIDTH = total representation width (the "10 format")
N_BITS = 4
REP_WIDTH = 10


# ============================================================
#   TRIT REPRESENTATION HELPERS
# ============================================================

def twos_complement_(uvec_lsb: list[int], width: int) -> list[int]:
    """LSB-first two's complement for uncertain-binary trits."""
    full_vec = uvec_lsb + [-1] * (width - len(uvec_lsb))

    out = []
    found_one = False

    for t in full_vec:
        if not found_one:
            out.append(+1 if t == +1 else -1 if t == -1 else 0)
            if t == +1:
                found_one = True
        else:
            if t == +1:
                out.append(-1)
            elif t == -1:
                out.append(+1)
            else:
                out.append(0)

    return out


def trits_to_decimal_bounds(bits):
    """
    Compute (min_value, max_value) bounds for a *positive* word.

    Semantics:
        1  -> definitely contributes 2^i
        -1 -> definitely contributes 0
        0  -> uncertain in [0, 2^i]

    We use the full REP_WIDTH word (bits[0] LSB).
    """
    mn = 0
    mx = 0
    for i, t in enumerate(bits):
        if t == 1:
            mn += (1 << i)
            mx += (1 << i)
        elif t == 0:
            mx += (1 << i)
    return mn, mx


def adder_sum_trits(adder_func, A, B):
    """Apply a ternary full adder bitwise over two 10-trit words."""
    carry = -1
    res = []
    for i in range(len(A)):
        s, carry = adder_func(A[i], B[i], carry)
        res.append(s)
    if carry != -1:
        res.append(carry)
    return res


def min_value(adder_func, A, B):
    """Lower bound of numeric value from adder output."""
    s = adder_sum_trits(adder_func, A, B)
    v = 0
    for i, t in enumerate(s):
        # 1 is definitely 1, everything else is 0 for the min
        v += ((1 if t == 1 else 0) << i)
    return v


def max_value(adder_func, A, B):
    """Upper bound of numeric value from adder output."""
    s = adder_sum_trits(adder_func, A, B)
    v = 0
    for i, t in enumerate(s):
        # -1 is definitely 0, 0 or 1 may contribute
        v += ((1 if t != -1 else 0) << i)
    return v


# ============================================================
#   GENERATE UNSIGNED (POSITIVE) VECTORS IN WIDTH-10 FORMAT
# ============================================================

def gen_unsigned_vectors(n_bits=N_BITS, width=REP_WIDTH):
    """
    Generate all *positive* magnitude patterns:
    - First n_bits trits explore all {-1,0,1}
    - Remaining bits (width - n_bits) are padded with -1 (definite 0)
    """
    choices = [-1, 0, 1]
    for core in itertools.product(choices, repeat=n_bits):
        core = list(core)
        padding = [-1] * (width - n_bits)
        yield core + padding


# ============================================================
#   MAIN EXPERIMENT (WITH POSITIVE & NEGATIVE A, B)
# ============================================================

results = {}

# All positive base patterns in width-10
base_vecs = list(gen_unsigned_vectors())
num_base = len(base_vecs)
print(f"Base positive patterns: {num_base}")

# accumulators per adder
acc = {name: {"min": [], "max": [], "mid": []} for name in adders}

# Loop over unordered base pairs (A_core, B_core)
for idx_a, A_core in enumerate(base_vecs):
    # Bounds for +A
    A_pos_min, A_pos_max = trits_to_decimal_bounds(A_core)

    for idx_b in range(idx_a, num_base):
        B_core = base_vecs[idx_b]
        B_pos_min, B_pos_max = trits_to_decimal_bounds(B_core)

        # Four sign combinations: (+A,+B), (+A,-B), (-A,+B), (-A,-B)
        for sign_a in (+1, -1):
            if sign_a == 1:
                A_bits = A_core
                A_min = A_pos_min
                A_max = A_pos_max
            else:
                A_bits = twos_complement_(A_core, width=REP_WIDTH)
                A_min = -B_pos_max  # careful: we negate *A* magnitude
                A_min = -A_pos_max
                A_max = -A_pos_min

            for sign_b in (+1, -1):
                if sign_b == 1:
                    B_bits = B_core
                    B_min = B_pos_min
                    B_max = B_pos_max
                else:
                    B_bits = twos_complement_(B_core, width=REP_WIDTH)
                    B_min = -B_pos_max
                    B_max = -B_pos_min

                true_min = A_min + B_min
                true_max = A_max + B_max
                true_mid = 0.5 * (true_min + true_max)

                # Evaluate each adder
                for name, func in adders.items():
                    a_min = min_value(func, A_bits, B_bits)
                    a_max = max_value(func, A_bits, B_bits)
                    a_mid = 0.5 * (a_min + a_max)

                    acc[name]["min"].append(abs(a_min - true_min))
                    acc[name]["max"].append(abs(a_max - true_max))
                    acc[name]["mid"].append(abs(a_mid - true_mid))

# Summarize stats
for name, d in acc.items():
    results[name] = {}
    for key in ("min", "max", "mid"):
        arr = np.array(d[key], dtype=float)
        results[name][key] = {
            "count": int(arr.size),
            "avg": float(arr.mean()) if arr.size else None,
            "max": float(arr.max()) if arr.size else None,
            "var": float(arr.var()) if arr.size else None,
            "std": float(arr.std()) if arr.size else None,
        }

out_file = "../uncertainty_results_signed_10bits.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {out_file}")
