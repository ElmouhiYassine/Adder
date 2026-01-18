
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

def luka_and(a: int, b: int) -> int:
    return LUKA_AND_TABLE[(a, b)]


def luka_or(a: int, b: int) -> int:
    return LUKA_OR_TABLE[(a, b)]

def luka_xor(a: int, b: int) -> int:
    return luka_and(luka_or(a, b), -luka_and(a, b))

def luka_add(a: int, b: int, carry: int):

    ab_xor = luka_xor(a, b)

    sum_digit = luka_xor(ab_xor, carry)

    gen = luka_and(a, b)
    prop = luka_and(carry, ab_xor)
    carry_out = luka_or(gen, prop)

    return sum_digit, carry_out

def luka_ripple_add(vecs):

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
            s, local_carry = luka_add(a, b, local_carry)
            new_result.append(s)

        result = new_result
        carry = local_carry


    if carry != -1:
        result.append(carry)

    return result
