
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

def gaines_and(a: int, b: int) -> int:
    return GAINES_AND_TABLE[(a, b)]


def gaines_or(a: int, b: int) -> int:
    return GAINES_OR_TABLE[(a, b)]


def gaines_not(x: int) -> int:
    return -x


def gaines_xor(a: int, b: int) -> int:
    ab_or = gaines_or(a, b)
    ab_and = gaines_and(a, b)
    not_and = gaines_not(ab_and)
    return gaines_and(ab_or, not_and)

def gaines_add(a: int, b: int, carry: int):

    ab_xor = gaines_xor(a, b)
    sum_digit = gaines_xor(ab_xor, carry)

    gen = gaines_and(a, b)
    prop = gaines_and(carry, ab_xor)
    carry_out = gaines_or(gen, prop)

    return sum_digit, carry_out


def gaines_ripple_add(vecs):

    if not vecs:
        return [-1] * 4

    result = vecs[0].copy()
    carry = -1

    for vec in vecs[1:]:


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


    if carry != -1:
        result.append(carry)

    return result
