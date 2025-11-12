import itertools
import numpy as np

# Import only the three adder implementations we care about:
from Adders.pesi_op_adder import full_op_adder
from Adders.SK_Quasi_adder import map_quasi_adder
from Adders.strong_kleene import strong_kleene_full_adder

from Adders.Ternary_New_Adder import (
    get_Adder
)

adder1 = get_Adder(0)
adder_collab= get_Adder(2)
adder_tr = get_Adder(4)
adder_2_tr = get_Adder(5)

def adder_sum_trits(adder_func, A: list[int], B: list[int]) -> list[int]:

    carry = -1
    result = []
    for i in range(len(A)):
        s, carry = adder_func(A[i], B[i], carry)
        result.append(s)
    if carry != -1:
        result.append(carry)
    return result

def trits_to_decimal_bounds(bits: list[int]) -> tuple[int,int]:

    min_val = 0
    max_val = 0
    for i, t in enumerate(bits):
        if t == +1:
            min_val += (1 << i)
            max_val += (1 << i)
        elif t == -1:
            # contributes 0 in both cases
            continue
        else:  # t == 0 → uncertain
            # min contribution = 0, max contribution = 1<<i
            max_val += (1 << i)
    return min_val, max_val

def min_value(adder_func, A: list[int], B: list[int]) -> int:
    sum_trits = adder_sum_trits(adder_func, A, B)
    value = 0
    for i, t in enumerate(sum_trits):
        bit = 1 if t == 1 else 0
        value += (bit << i)
    return value

def max_value(adder_func, A: list[int], B: list[int]) -> int:

    sum_trits = adder_sum_trits(adder_func, A, B)
    value = 0
    for i, t in enumerate(sum_trits):
        bit = 1 if t != -1 else 0
        value += (bit << i)
    return value


adders = {
    "Strong-Kleene": strong_kleene_full_adder,
    "Pessimistic-Op": full_op_adder,
    "Quasi": map_quasi_adder,
    "Super Pessi-Op" : adder1,
    "Collapsible" : adder_collab,
    "Triangular" : adder_tr,
    "Bi-Triangular" : adder_2_tr,
}

# vectors = [v for v in itertools.product([ -1,0], repeat=8) if is_mu_vector(v)]
vectors = list(itertools.product([-1, 0], repeat=8))

# Prepare a stats table for each adder with lists for variance calculation
# Prepare a stats table for each adder with lists for variance calculation
stats = {}
for name in adders:
    stats[name] = {
        "min_dev": [],
        "max_dev": [],
        "mid_dev": [],
        "min_pct": [],
        "max_pct": [],
        "mid_pct": [],
    }

count = 0

for A, B in itertools.combinations_with_replacement(vectors, 2):
    A = list(A); B = list(B)
    count += 1
    # True bounds
    A_min, A_max = trits_to_decimal_bounds(A)
    B_min, B_max = trits_to_decimal_bounds(B)
    true_min = A_min + B_min
    true_max = A_max + B_max
    true_mid = 0.5 * (true_min + true_max)

    for name, func in adders.items():
        adder_min = min_value(func, list(A), list(B))
        adder_max = max_value(func, list(A), list(B))
        adder_mid = 0.5 * (adder_min + adder_max)

        # Absolute deviations
        stats[name]["min_dev"].append(abs(adder_min - true_min))
        stats[name]["max_dev"].append(abs(adder_max - true_max))
        stats[name]["mid_dev"].append(abs(adder_mid - true_mid))

        # Relative errors in percentage
        if true_min == 0:
            if adder_min == 0:
                stats[name]["min_pct"].append(0.0)
            else:
                # If truth is 0 but adder is not, treat error as full magnitude ratio to 1 (or set rule)
                stats[name]["min_pct"].append(100.0)
        else:
            stats[name]["min_pct"].append(100 * abs(adder_min - true_min) / abs(true_min))

        if true_max == 0:
            if adder_max == 0:
                stats[name]["max_pct"].append(0.0)
            else:
                stats[name]["max_pct"].append(100.0)
        else:
            stats[name]["max_pct"].append(100 * abs(adder_max - true_max) / abs(true_max))

        if true_mid == 0:
            if adder_mid == 0:
                stats[name]["mid_pct"].append(0.0)
            else:
                stats[name]["mid_pct"].append(100.0)
        else:
            stats[name]["mid_pct"].append(100 * abs(adder_mid - true_mid) / abs(true_mid))

print(count)

# Print results
for name, data in stats.items():
    min_dev = np.array(data["min_dev"])
    max_dev = np.array(data["max_dev"])
    mid_dev = np.array(data["mid_dev"])

    min_pct = np.array(data["min_pct"])
    max_pct = np.array(data["max_pct"])
    mid_pct = np.array(data["mid_pct"])

    print(f"Adder: {name}")
    print(f"  Avg abs(min error) = {min_dev.mean():.4f}   Var = {min_dev.var():.4f}   Avg % err = {min_pct.mean():.2f}%")
    print(f"  Avg abs(max error) = {max_dev.mean():.4f}   Var = {max_dev.var():.4f}   Avg % err = {max_pct.mean():.2f}%")
    print(f"  Avg abs(mid error) = {mid_dev.mean():.4f}   Var = {mid_dev.var():.4f}   Avg % err = {mid_pct.mean():.2f}%")
    print()