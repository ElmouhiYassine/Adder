import numpy as np


def balanced_ternary_add(A, B):

    # Determine the maximum length
    n = max(len(A), len(B))
    carry = 0
    result = []

    for i in range(n):
        # Get digits at current position (0 if beyond list length)
        a = A[i] if i < len(A) else 0
        b = B[i] if i < len(B) else 0

        # Calculate total sum
        total = a + b + carry

        # Compute carry and digit
        carry = (total + 1) // 3  # Integer division
        digit = total - 3 * carry

        result.append(digit)

    # Add final carry if non-zero
    if carry != 0:
        result.append(carry)

    return result

def BT_ripple_add(vecs):
    """Ripple-carry sum of list of trit vectors, returns final trit vector."""
    if not vecs:
        return [0] * 4

    result = vecs[0].copy()

    for vec in vecs[1:]:
        new_result = balanced_ternary_add(vec, result)
        result = new_result

    return result

# Test with your example
A = [1, -1, 1]  # 1*3⁰ + (-1)*3¹ + 1*3² = 1 - 3 + 9 = 7
B = [1, 0, -1]  # 1*3⁰ + 0*3¹ + (-1)*3² = 1 + 0 - 9 = -8

sum_result = balanced_ternary_add(A, B)
# print("Sum in balanced ternary (LSB first):", sum_result)
#
# # Convert back to decimal to verify
# decimal_value = 0
# for i, digit in enumerate(sum_result):
#     decimal_value += digit * (3 ** i)
# print("Decimal value:", decimal_value)

