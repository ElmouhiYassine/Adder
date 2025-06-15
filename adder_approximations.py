import itertools
import numpy as np

# Import only the three adder implementations we care about:
from pesi_op_adder import full_op_adder
from SK_Quasi_adder import map_quasi_adder
from strong_kleene import strong_kleene_full_adder

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
    "Quasi": map_quasi_adder
}

# Enumerate all 4-trit vectors in {-1,0,+1}^4:
vectors = list(itertools.product([-1, 0, +1], repeat=4))

# Prepare a stats table for each adder
stats = {}
for name in adders:
    stats[name] = {
        "sum_min_deviation": 0.0,   # sum of (|adder_min - true_min|)
        "sum_max_deviation": 0.0,   # sum of (|adder_max - true_max|)
        "sum_mid_deviation": 0.0,   # sum of (|adder_mid - true_mid|)
        "total_pairs": 0
    }

count = 0

# Loop over all pairs (A, B) in {-1,0,+1}^4 × {-1,0,+1}^4:
for A in vectors:
    for B in vectors:
        count += 1
        # Compute true‐operand bounds:
        A_min, A_max = trits_to_decimal_bounds(A)
        B_min, B_max = trits_to_decimal_bounds(B)
        true_min = A_min + B_min
        true_max = A_max + B_max
        true_mid = 0.5 * (true_min + true_max)

        for name, func in adders.items():
            adder_min = min_value(func, list(A), list(B))
            adder_max = max_value(func, list(A), list(B))
            adder_mid = 0.5 * (adder_min + adder_max)

            data = stats[name]
            data["total_pairs"] += 1
            data["sum_min_deviation"] += abs(adder_min - true_min)
            data["sum_max_deviation"] += abs(adder_max - true_max)
            data["sum_mid_deviation"] += abs(adder_mid - true_mid)

# Print out averaged metrics for each adder
for name, data in stats.items():
    n = data["total_pairs"]
    avg_min_dev = data["sum_min_deviation"] / n
    avg_max_dev = data["sum_max_deviation"] / n
    avg_mid_dev = data["sum_mid_deviation"] / n

    print(f"Adder: {name}")
    print(f"  Avg |adder_min - true_min| = {avg_min_dev:.4f}")
    print(f"  Avg |adder_max - true_max| = {avg_max_dev:.4f}")
    print(f"  Avg |adder_mid - true_mid| = {avg_mid_dev:.4f}")
    print()
