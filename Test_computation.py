from itertools import product
from Adders.Ternary_New_Adder import *
adder_tr = get_Adder(5)
def int_to_4bit(val):
    """Convert integer 0-15 to list of 4 bits (MSB first)."""
    return [(val >> i) & 1 for i in reversed(range(4))]

def encode_to_trits(val, noise_mask):
    """
    Encode val (0-15) into a 4-trit list with uncertainty mask.
    noise_mask: list of length 4 of booleans; True -> uncertain (0), else mapped from bit.
    """
    bits = int_to_4bit(val)
    trits = []
    for i, b in enumerate(bits):
        if noise_mask[i]:
            trits.append(0)
        else:
            trits.append(1 if b == 1 else -1)
    return trits

def ripple_add(vecs):
    """Ripple-carry sum of list of trit vectors, returns final trit vector."""
    if not vecs:
        return [-1] * 4

    result = vecs[0].copy()
    carry = -1  # neutral

    for vec in vecs[1:]:
        new_result = []
        for a, b in zip(result, vec):
            s, carry = adder_tr(a, b, carry)
            new_result.append(s)
        result = new_result

    # append the final carry
    if carry != -1:
        result.append(carry)

    return result

# Prepare noise masks: for k uncertain LSBs -> positions 4-k..3
# Generate all 4-length combinations of True/False
masks = list(product([False, True], repeat=4))
# Storage of all data
all_data = []
seen_pairs = set()

from itertools import product

As = list(product([1, 0, -1], repeat=4))
Bs = list(product([1, 0, -1], repeat=4))


for A in As:
    for B in Bs:
        key = tuple(sorted((A, B)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        result = ripple_add([list(A), list(B)])  # <-- convert here
        all_data.append({
            'A_trits': list(A),
            'B_trits': list(B),
            'sum': result
        })

# print(len(all_data))
#
# # Sample 100 random examples to inspect
# # Take 10 entries from each block of 100, up to 100 samples total
# samples = []
# for i in range(0, len(all_data), 100):
#     block = all_data[i:i + 100]
#     samples.extend(block[:10])
#     if len(samples) >= 100:
#         break
# text = """
# Sample & A & B & Sum & Observation \\\\
# \\hline
# """
# # Sample LaTeX row
# def format_row(sample_number, a, b, s):
#     return f"Sample {sample_number} & {a} & {b} & {s} & \\\\\n"
# # Add all rows to LaTeX
# for i, sample in enumerate(samples[:100], start=1):
#     a = sample['A_trits'][::-1]
#     b = sample['B_trits'][::-1]
#     s = sample['sum'][::-1]
#     text += format_row(i, a, b, s)
#
# print(text)
# Example visualization or printing
# for i, sample in enumerate(samples[:100]):
#     print(f"Sample {i+1}")
#     print(f" A: {sample['A_trits']}")
#     print(f" B: {sample['B_trits']}")
#     print(f" Sum: {sample['sum']}")
#     print("")

# You can export the full `samples` list to a CSV or use matplotlib for plotting if needed.
# print(list(reversed(ripple_add([list(reversed([-1,1,-1,1,-1,1,0,-1])), list(reversed([-1,1,-1,1,-1,0,0,0])) ]))))