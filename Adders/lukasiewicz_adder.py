# ============================================================
#      LUKASIEWICZ TERNARY LOGIC GATES (FROM YOUR TABLE)
# ============================================================

LUKA_AND_TABLE = {
    (1, 0): 0,
    (0, -1): -1,
    (-1, 0): -1,
    (0, 1): 0,
    (1, 1): 1,
    (-1, 1): -1,
    (0, 0): -1,
    (-1, -1): -1,
    (1, -1): -1
}

LUKA_OR_TABLE = {
    (0, 1): 1,
    (-1, 1): 1,
    (1, 1): 1,
    (1, 0): 1,
    (0, 0): 1,
    (0, -1): 0,
    (-1, 0): 0,
    (-1, -1): -1,
    (1, -1): 1
}

# ============================================================
# 1) LUKASIEWICZ AND
# ============================================================

def luka_and(a: int, b: int) -> int:
    return LUKA_AND_TABLE[(a, b)]

# ============================================================
# 2) LUKASIEWICZ OR
# ============================================================

def luka_or(a: int, b: int) -> int:
    return LUKA_OR_TABLE[(a, b)]

# ============================================================
# 3) LUKASIEWICZ XOR
# ============================================================
# We define XOR as:
#  XOR(a,b) = OR(a,b) when a != b, else AND(a,b)
# This is consistent with ternary variational XOR.

def luka_xor(a: int, b: int) -> int:
    if a == b:
        return luka_and(a, b)
    return luka_or(a, b)

# ============================================================
# 4) LUKASIEWICZ FULL ADDER FOR ONE DIGIT
# ============================================================
# Inputs: a, b, carry (each in {-1, 0, 1})
# Output: (sum_digit, next_carry)

def luka_add(a: int, b: int, carry: int):
    """
    Lukasiewicz full adder:
       sum = XOR(XOR(a, b), carry)
       carry_out = OR(AND(a,b), AND(carry, XOR(a,b)))
    """

    # First XOR stage
    ab_xor = luka_xor(a, b)

    # Sum digit
    sum_digit = luka_xor(ab_xor, carry)

    # Carry logic
    gen = luka_and(a, b)            # generate
    prop = luka_and(carry, ab_xor)  # propagate
    carry_out = luka_or(gen, prop)

    return sum_digit, carry_out

# ============================================================
# 5) LUKASIEWICZ RIPPLE ADDER FOR MULTIPLE VECTORS
# ============================================================

def luka_ripple_add(vecs):
    """
    Ripple addition of many ternary vectors (LSB-first),
    using Lukasiewicz ternary gates.
    """

    if not vecs:
        return [-1] * 4  # neutral fallback

    # Start with the first vector
    result = vecs[0].copy()
    carry = -1  # neutral carry for ternary logic

    # Add the remaining vectors one by one
    for vec in vecs[1:]:

        # Ensure equal length
        L = max(len(result), len(vec))
        A = result + [0] * (L - len(result))
        B = vec + [0] * (L - len(vec))

        new_result = []
        local_carry = carry

        for a, b in zip(A, B):
            s, local_carry = luka_add(a, b, local_carry)
            new_result.append(s)

        result = new_result
        carry = local_carry

    # Append final carry if non-neutral
    if carry != -1:
        result.append(carry)

    return result
