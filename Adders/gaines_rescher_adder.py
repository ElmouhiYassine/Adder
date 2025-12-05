"""
Gaines–Rescher Logic – AND, OR, XOR, Full Adder, Ripple Adder
Compatible with HTL ternary convolution system.
Trits are in {-1, 0, 1}.
"""

# ============================================================
#                  GAINES–RESCHER TRUTH TABLES
# ============================================================

GAINES_AND_TABLE = {
    (0, 1): 1,
    (0, -1): -1,
    (-1, 0): -1,
    (1, 1): 1,
    (-1, 1): -1,
    (1, 0): 1,
    (0, 0): -1,
    (-1, -1): -1,
    (1, -1): -1,
}

GAINES_OR_TABLE = {
    (0, 1): 1,
    (0, -1): -1,
    (-1, 0): -1,
    (-1, 1): 1,
    (1, 1): 1,
    (1, 0): 1,
    (0, 0): 1,
    (-1, -1): -1,
    (1, -1): 1,
}

# ============================================================
#                        LOGICAL GATES
# ============================================================

def gaines_and(a: int, b: int) -> int:
    """Gaines–Rescher AND gate."""
    return GAINES_AND_TABLE[(a, b)]


def gaines_or(a: int, b: int) -> int:
    """Gaines–Rescher OR gate."""
    return GAINES_OR_TABLE[(a, b)]


def gaines_not(x: int) -> int:
    """HTL negation rule: NOT(x) = -x."""
    return -x


def gaines_xor(a: int, b: int) -> int:
    """
    XOR(a,b) = (a OR b) AND NOT(a AND b)
    (Definition must use gates exactly; never invent rules.)
    """
    ab_or = gaines_or(a, b)
    ab_and = gaines_and(a, b)
    not_and = gaines_not(ab_and)
    return gaines_and(ab_or, not_and)

# ============================================================
#                     FULL ADDER (1 TRIT)
# ============================================================

def gaines_add(a: int, b: int, carry: int):
    """
    Full adder using Gaines–Rescher logic:

    sum = XOR(XOR(a,b), carry)
    carry_out = OR( AND(a,b), AND(carry, XOR(a,b)) )
    """

    ab_xor = gaines_xor(a, b)
    sum_digit = gaines_xor(ab_xor, carry)

    gen = gaines_and(a, b)
    prop = gaines_and(carry, ab_xor)
    carry_out = gaines_or(gen, prop)

    return sum_digit, carry_out

# ============================================================
#                 RIPPLE ADDER (MULTI-VECTOR)
# ============================================================

def gaines_ripple_add(vecs):
    """
    Ripple-add multiple ternary vectors (LSB-first)
    using Gaines–Rescher logic.
    """

    if not vecs:
        return [-1] * 4  # neutral fallback

    result = vecs[0].copy()
    carry = -1  # neutral HTL carry

    for vec in vecs[1:]:

        # Equalize lengths
        L = max(len(result), len(vec))
        A = result + [0] * (L - len(result))
        B = vec + [0] * (L - len(vec))

        new_res = []
        local_carry = carry

        for a, b in zip(A, B):
            s, local_carry = gaines_add(a, b, local_carry)
            new_res.append(s)

        result = new_res
        carry = local_carry

    # Append carry if non-neutral
    if carry != -1:
        result.append(carry)

    return result
