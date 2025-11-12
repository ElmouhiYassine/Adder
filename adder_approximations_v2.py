import itertools
import json
import numpy as np
import random

# ---- your adders (replace imports if needed) ----
from Adders.pesi_op_adder import full_op_adder
from Adders.SK_Quasi_adder import map_quasi_adder
from Adders.strong_kleene import strong_kleene_full_adder
from Adders.Ternary_New_Adder import get_Adder

adder1 = get_Adder(0)
adder_collab = get_Adder(2)
adder_tr = get_Adder(4)
adder_2_tr = get_Adder(5)

adders = {
    "Strong-Kleene": strong_kleene_full_adder,
    "Pessimistic-Op": full_op_adder,
    "Quasi": map_quasi_adder,
    "Super Pessi-Op": adder1,
    "Collapsible": adder_collab,
    "Triangular": adder_tr,
    "Bi-Triangular": adder_2_tr,
}
# -------------------------------------------------

N_BITS = 10
SAMPLES_PER_LEVEL = 200000   # number of (A,B) samples to test per level
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def trits_to_decimal_bounds(bits):
    mn = 0; mx = 0
    for i, t in enumerate(bits):
        if t == 1:
            mn += (1 << i); mx += (1 << i)
        elif t == 0:
            mx += (1 << i)
    return mn, mx

def adder_sum_trits(adder_func, A, B):
    carry = -1; res = []
    for i in range(len(A)):
        s, carry = adder_func(A[i], B[i], carry)
        res.append(s)
    if carry != -1:
        res.append(carry)
    return res

def min_value(adder_func, A, B):
    s = adder_sum_trits(adder_func, A, B)
    v = 0
    for i,t in enumerate(s):
        v += ((1 if t==1 else 0) << i)
    return v

def max_value(adder_func, A, B):
    s = adder_sum_trits(adder_func, A, B)
    v = 0
    for i,t in enumerate(s):
        v += ((1 if t!=-1 else 0) << i)
    return v

def gen_vectors_for_level(level):
    choices = [([-1,0,1] if i < level else [-1,1]) for i in range(N_BITS)]
    return list(itertools.product(*choices))

results = {}
for level in range(N_BITS + 1):
    vecs = gen_vectors_for_level(level)
    V = len(vecs)
    print(f"Level {level}: {V} vectors (sampling {SAMPLES_PER_LEVEL} pairs)")

    acc = {name: {"min": [], "max": [], "mid": []} for name in adders}

    # Directly sample without building the huge list of pairs
    sampled_pairs = set()
    while len(sampled_pairs) < min(SAMPLES_PER_LEVEL, V * (V + 1) // 2):
        A = random.choice(vecs)
        B = random.choice(vecs)
        # enforce combinations_with_replacement behaviour
        if A > B:
            A, B = B, A
        sampled_pairs.add((A, B))

    for A, B in sampled_pairs:
        A = list(A); B = list(B)
        A_min, A_max = trits_to_decimal_bounds(A)
        B_min, B_max = trits_to_decimal_bounds(B)
        true_min = A_min + B_min
        true_max = A_max + B_max
        true_mid = 0.5 * (true_min + true_max)

        for name, func in adders.items():
            a_min = min_value(func, A, B)
            a_max = max_value(func, A, B)
            a_mid = 0.5 * (a_min + a_max)

            acc[name]["min"].append(abs(a_min - true_min))
            acc[name]["max"].append(abs(a_max - true_max))
            acc[name]["mid"].append(abs(a_mid - true_mid))

    # summarize
    results[level] = {}
    for name, d in acc.items():
        results[level][name] = {}
        for key in ("min", "max", "mid"):
            arr = np.array(d[key], dtype=float)
            results[level][name][key] = {
                "count": int(arr.size),
                "avg": float(arr.mean()) if arr.size else None,
                "max": float(arr.max()) if arr.size else None,
                "var": float(arr.var()) if arr.size else None,
                "std": float(arr.std()) if arr.size else None
            }

with open("uncertainty_results_vectors_10_trits_sampled.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved uncertainty_results_vectors_10_trits_sampled.json")
