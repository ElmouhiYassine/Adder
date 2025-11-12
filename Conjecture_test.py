# import itertools
# from strong_kleene import strong_kleene_full_adder
#
# def vector_add(acc: list[int], v: int) -> list[int]:
#     """
#     Ajoute un trit `v` à un vecteur accumulateur `acc` de 4 trits
#     en utilisant le full adder strong kleene.
#     acc[0] est le LSB.
#     """
#     carry = -1
#     result = []
#     for bit in acc:
#         s, carry = strong_kleene_full_adder(bit, v, carry)
#         result.append(s)
#     return result
#
# def strong_kleene_vector_sum(window: tuple[int, ...]) -> list[int]:
#     """
#     Fait la somme des 8 trits d'une fenêtre via accumulation binaire sur 4 trits.
#     """
#     acc = [-1, -1, -1, -1]  # 4-bit accumulator
#     for v in window:
#         acc = vector_add(acc, v)
#     return acc
#
# # Conjecture: si l'entrée contient une incertitude (0),
# # alors la sortie doit aussi contenir au moins un 0.
# bad_cases = []
# for vec in itertools.product([-1, 0, 1], repeat=15):
#     if any(v == 0 for v in vec):  # si entrée incertaine
#         result = strong_kleene_vector_sum(vec)
#         if all(t != 0 for t in result):  # mais aucune incertitude dans la sortie
#             bad_cases.append((vec, result))
#
# # Affichage du résultat
# if not bad_cases:
#     print("✔️  Conjecture holds: any uncertain input → uncertain output (0 present).")
# else:
#     print(f"❌  Found {len(bad_cases)} counter-example(s). First one:")
#     window, out = bad_cases[0]
#     print("   Input  :", window)
#     print("   Output :", out)

import itertools
from Adders.SK_Quasi_adder import map_quasi_adder


def vector_add(acc: list[int], v: int) -> list[int]:
    carry = -1
    out = []
    for bit in acc:
        s, carry = map_quasi_adder(bit, v, carry)
        out.append(s)
    out.append(carry)
    return out


def pessimistic_vector_sum(window: tuple[int, ...]) -> list[int]:
    """
    Fold an 8‐tuple of values through a 4‑bit accumulator.
    Start with [-1,-1,-1,-1], then for each v in window do:
       acc = vector_add(acc, v)
    """
    acc = [-1, -1, -1, -1]
    for v in window:
        acc = vector_add(acc, v)
    return acc

# Collect counter‑examples:
# Conjecture test: unlikely to get uncertainty in output
bad = []
for window in itertools.product([-1, 0, 1], repeat=15):
    out_vec = pessimistic_vector_sum(window)
    # ❌ If any output value is 0 → uncertainty → counterexample
    if any(v == 0 for v in out_vec):
        bad.append((window, out_vec))
        break  # one counterexample is enough

if not bad:
    print("✔️  Conjecture holds: No uncertainty (0) in output for any 16-input combination.")
else:
    window, result = bad[0]
    print("❌  Conjecture fails!")
    print("Input window:", window)
    print("Output vec  :", result)
