"""
Bochvar External Logic – AND, OR, XOR, Full Adder, Ripple Adder
Compatible with HTL convolution framework (LSB-first ternary vectors).
Trits are in {-1, 0, 1}.
"""

# ============================================================
#                BOCHVAR EXTERNAL TRUTH TABLES
# ============================================================

BOCHVAR_AND_TABLE = {
    (0, 1): -1,
    (0, -1): -1,
    (-1, 0): -1,
    (1, 1): 1,
    (-1, 1): -1,
    (1, 0): -1,
    (0, 0): -1,
    (-1, -1): -1,
    (1, -1): -1,
}

BOCHVAR_OR_TABLE = {
    (0, -1): 1,
    (0, 1): 1,
    (-1, 1): 1,
    (-1, 0): 1,
    (1, 1): 1,
    (1, 0): 1,
    (0, 0): 1,
    (-1, -1): -1,
    (1, -1): 1,
}

# ============================================================
#                     LOGICAL GATES
# ============================================================

def bochvar_and(a: int, b: int) -> int:
    """Bochvar external AND gate."""
    return BOCHVAR_AND_TABLE[(a, b)]


def bochvar_or(a: int, b: int) -> int:
    """Bochvar external OR gate."""
    return BOCHVAR_OR_TABLE[(a, b)]


def bochvar_not(x: int) -> int:
    """
    Negation = -x  (your rule)
        1 -> -1
        0 -> 0
        -1 -> 1
    """
    return -x


def bochvar_xor(a: int, b: int) -> int:
    """
    XOR(a,b) = (a OR b) AND NOT(a AND b)
    Using ONLY Bochvar logic operations.
    """
    ab_or = bochvar_or(a, b)
    ab_and = bochvar_and(a, b)
    not_and = bochvar_not(ab_and)
    return bochvar_and(ab_or, not_and)


# ============================================================
#                   FULL ADDER (ONE DIGIT)
# ============================================================

def bochvar_add(a: int, b: int, carry: int):
    """
    One-digit full adder using Bochvar XOR, AND, OR.

    sum_digit = XOR(XOR(a,b), carry)
    carry_out = OR( AND(a,b), AND(carry, XOR(a,b)) )
    """

    ab_xor = bochvar_xor(a, b)
    sum_digit = bochvar_xor(ab_xor, carry)

    gen = bochvar_and(a, b)
    prop = bochvar_and(carry, ab_xor)
    carry_out = bochvar_or(gen, prop)

    return sum_digit, carry_out


# ============================================================
#               MULTI-VECTOR RIPPLE ADDER (LSB-FIRST)
# ============================================================

def bochvar_ripple_add(vecs):
    """
    Ripple-adds multiple ternary vectors (LSB-first)
    using Bochvar external logic.

    Each vector is a list of trits in {-1,0,1}.
    """

    if not vecs:
        return [-1] * 4  # neutral fallback

    result = vecs[0].copy()
    carry = -1  # neutral carry

    for vec in vecs[1:]:

        # Pad both vectors to same length
        L = max(len(result), len(vec))
        A = result + [0] * (L - len(result))
        B = vec + [0] * (L - len(vec))

        new_result = []
        local_carry = carry

        for a, b in zip(A, B):
            s, local_carry = bochvar_add(a, b, local_carry)
            new_result.append(s)

        result = new_result
        carry = local_carry

    # Append carry if non-neutral
    if carry != -1:
        result.append(carry)

    return result
