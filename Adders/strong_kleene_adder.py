

def kleene_and(a, b):

    if a == -1 or b == -1:
        return -1
    if a == 1 and b == 1:
        return 1
    return 0


def kleene_or(a, b):

    if a == 1 or b == 1:
        return 1
    if a == -1 and b == -1:
        return -1
    return 0


def kleene_xor(a, b):

    return kleene_and(kleene_or(a,b),-kleene_and(a,b))


def strong_kleene_full_adder(a, b, cin = -1):

    s1 = kleene_xor(a, b)
    sum_bit = kleene_xor(s1, cin)

    ab = kleene_and(a, b)

    carry_out = kleene_or(ab,kleene_and(cin,s1))

    return sum_bit, carry_out

def SK_ripple_add(vecs):
    if not vecs:
        return [-1] * 4

    result = vecs[0].copy()
    carry = -1

    for vec in vecs[1:]:
        new_result = []
        for a, b in zip(result, vec):
            s, carry = strong_kleene_full_adder(a, b, carry)
            new_result.append(s)


        result = new_result

    if carry != -1:
        result.append(carry)

    return result

