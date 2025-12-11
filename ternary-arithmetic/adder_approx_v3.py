import itertools
import json
import numpy as np
import matplotlib.pyplot as plt

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
# ---------------------

N_BITS = 8
REP_WIDTH = 10
SIGN_INDEX = REP_WIDTH - 1


def twos_complement_(vec, width):
    vec = vec[:] + [-1] * (width - len(vec))
    result = []
    found_one = False
    for t in vec:
        if not found_one:
            if t == +1:
                result.append(+1)
                found_one = True
            elif t == -1:
                result.append(-1)
            else:
                result.append(0)
        else:
            if t == +1:
                result.append(-1)
            elif t == -1:
                result.append(+1)
            else:
                result.append(0)
    return result


def magnitude_bounds(bits):
    mn = 0
    mx = 0
    for i, t in enumerate(bits):
        if t == 1:
            mn += (1 << i)
            mx += (1 << i)
        elif t == 0:
            mx += (1 << i)
    return mn, mx


def decode_bounds_from_trits(trits):
    mn = 0
    mx = 0

    # bits 0..8 behave normally
    for i, t in enumerate(trits[:-1]):   # up to SIGN_INDEX-1
        w = (1 << i)
        if t == 1:
            mn += w
            mx += w
        elif t == 0:
            mx += w
        # t == -1 → contributes nothing

    # handle sign trit
    s = trits[SIGN_INDEX]
    w = (1 << SIGN_INDEX)    # weight of the sign bit (e.g., 512)

    if s == 1:
        # definitely negative contribution
        mn -= w
        mx -= w
    elif s == 0:
        # uncertain between 0 and -2^SIGN_INDEX
        # min includes negative contribution
        mn -= w
        # max = no contribution (0)
        # so mx unchanged
        pass
    elif s == -1:
        pass

    return mn, mx


def adder_sum_trits(adder_func, A, B):
    carry = -1
    out = []
    for i in range(REP_WIDTH):
        s, carry = adder_func(A[i], B[i], carry)
        out.append(s)
    return out[:REP_WIDTH]


def adder_output_bounds(adder_func, A, B):
    out = adder_sum_trits(adder_func, A, B)
    return decode_bounds_from_trits(out)


def min_value(adder_func, A, B):
    mn, _ = adder_output_bounds(adder_func, A, B)
    return mn


def max_value(adder_func, A, B):
    _, mx = adder_output_bounds(adder_func, A, B)
    return mx


# ============================================================
# GENERATE VALID WORDS FOR A GIVEN UNCERTAINTY LEVEL
# ============================================================
def gen_unsigned_vectors(level, n_bits=N_BITS, width=REP_WIDTH):
    """
    level = maximum allowed uncertainty depth.
    Words of the form:
        [0..0][±1 .. ±1]   with prefix length ≤ level
    """
    for k in range(level + 1):  # prefix of 0's up to 'level'
        prefix = [0] * k
        for tail in itertools.product([-1, 1], repeat=n_bits - k):
            word = prefix + list(tail)
            padded = word + [-1] * (width - n_bits)
            yield padded



LEVELS = list(range(0, N_BITS + 1))  # 0..8
level_results = {adder: {"min": [], "max": [], "mid": []} for adder in adders}

for lvl in LEVELS:
    print(f"\n=== Testing uncertainty level {lvl} ===")

    base_vecs = list(gen_unsigned_vectors(lvl))
    num_base = len(base_vecs)
    print(f"Valid vectors at this level: {num_base}")

    # accumulate error for this level (per adder)
    acc = {name: {"min": [], "max": [], "mid": []} for name in adders}

    for i, A_core in enumerate(base_vecs):
        A_pos_min, A_pos_max = magnitude_bounds(A_core[:SIGN_INDEX])

        for j in range(i, num_base):
            B_core = base_vecs[j]
            B_pos_min, B_pos_max = magnitude_bounds(B_core[:SIGN_INDEX])

            # Four sign configurations
            for sign_a in (+1, -1):
                if sign_a == +1:
                    A_bits = A_core[:]
                    A_min, A_max = A_pos_min, A_pos_max
                else:
                    A_bits = twos_complement_(A_core, REP_WIDTH)
                    A_min, A_max = -A_pos_max, -A_pos_min

                for sign_b in (+1, -1):
                    if sign_b == +1:
                        B_bits = B_core[:]
                        B_min, B_max = B_pos_min, B_pos_max
                    else:
                        B_bits = twos_complement_(B_core, REP_WIDTH)
                        B_min, B_max = -B_pos_max, -B_pos_min

                    true_min = A_min + B_min
                    true_max = A_max + B_max
                    true_mid = 0.5 * (true_min + true_max)

                    for name, func in adders.items():
                        a_min = min_value(func, A_bits, B_bits)
                        a_max = max_value(func, A_bits, B_bits)
                        a_mid = 0.5 * (a_min + a_max)

                        acc[name]["min"].append(abs(a_min - true_min))
                        acc[name]["max"].append(abs(a_max - true_max))
                        acc[name]["mid"].append(abs(a_mid - true_mid))

    # store average errors per level
    for name in adders:
        for metric in ["min", "max", "mid"]:
            arr = np.array(acc[name][metric], float)
            level_results[name][metric].append(arr.mean())



def plot_metric(metric):
    plt.figure(figsize=(10, 6))

    num_adders = len(adders)
    offset_range = 0.15
    offsets = np.linspace(-offset_range, offset_range, num_adders)

    for (idx, name), offset in zip(enumerate(adders), offsets):
        x_shifted = np.array(LEVELS) + offset

        plt.plot(
            x_shifted,
            level_results[name][metric],
            marker="o",
            label=name
        )

    plt.title(f"Average {metric} error vs Uncertainty Level")
    plt.xlabel("Uncertainty level in an 8-bit number")
    plt.ylabel(f"Mean absolute {metric} error")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    # 🔥 Force x-axis ticks to show every level 0..8
    plt.xticks(LEVELS)

    plt.tight_layout()
    plt.savefig(f"comparison_{metric}.png", dpi=300)
    plt.show()




plot_metric("min")
plot_metric("max")
plot_metric("mid")



with open("../uncertainty_results_by_level.json", "w") as f:
    json.dump(level_results, f, indent=2)

print("\nSaved: uncertainty_results_by_level.json")
