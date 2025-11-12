from Adders.Ternary_New_Adder import get_Adder
lukasiewicz_adder =get_Adder(0)


def lukasiewicz_ripple(vecs):
    if not vecs:
        return [-1] * 4

    result = vecs[0].copy()
    carry = -1  # neutral

    for vec in vecs[1:]:
        new_result = []
        for a, b in zip(result, vec):
            s, carry = lukasiewicz_adder(a, b, carry)
            new_result.append(s)
        result = new_result

    # append the final carry
    if carry != -1:
        result.append(carry)

    return result