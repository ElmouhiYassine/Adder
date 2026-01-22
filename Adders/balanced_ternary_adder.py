import numpy as np


def BT_add(A, B):

    n = max(len(A), len(B))
    carry = 0
    result = []

    for i in range(n):
        a = A[i] if i < len(A) else 0
        b = B[i] if i < len(B) else 0

        total = a + b + carry

        carry = (total + 1) // 3
        digit = total - 3 * carry

        result.append(digit)

    if carry != 0:
        result.append(carry)

    return result

def BT_ripple_add(vecs):
    if not vecs:
        return [0] * 4

    result = vecs[0].copy()

    for vec in vecs[1:]:
        new_result = BT_add(vec, result)
        result = new_result

    return result

# # example
# A = [1, -1, 1]  # 1*3⁰ + (-1)*3¹ + 1*3² = 1 - 3 + 9 = 7
# B = [1, 0, -1]  # 1*3⁰ + 0*3¹ + (-1)*3² = 1 + 0 - 9 = -8
#
# sum_result = balanced_ternary_add(A, B)
# print("Sum in balanced ternary (LSB first):", sum_result)
#

