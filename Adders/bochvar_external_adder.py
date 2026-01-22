
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

def bochvar_and(a: int, b: int) -> int:
    return BOCHVAR_AND_TABLE[(a, b)]


def bochvar_or(a: int, b: int) -> int:
    return BOCHVAR_OR_TABLE[(a, b)]


def bochvar_not(x: int) -> int:
    return -x


def bochvar_xor(a: int, b: int) -> int:

    ab_or = bochvar_or(a, b)
    ab_and = bochvar_and(a, b)
    not_and = bochvar_not(ab_and)
    return bochvar_and(ab_or, not_and)


def bochvar_add(a: int, b: int, carry: int):

    ab_xor = bochvar_xor(a, b)
    sum_digit = bochvar_xor(ab_xor, carry)

    gen = bochvar_and(a, b)
    prop = bochvar_and(carry, ab_xor)
    carry_out = bochvar_or(gen, prop)

    return sum_digit, carry_out


def bochvar_ripple_add(vecs):

    if not vecs:
        return [-1] * 4

    result = vecs[0].copy()
    carry = -1

    for vec in vecs[1:]:

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

    if carry != -1:
        result.append(carry)

    return result
