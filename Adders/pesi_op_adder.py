ternary_oor_table = {
    (1, 1): 1,
    (1, 0): 1,
    (1, -1): 1,
    (0, 1): 1,
    (0, 0): 0,
    (0, -1): 1,
    (-1, 1): 1,
    (-1, 0): 1,
    (-1, -1): -1
}

ternary_pand_table = {
    (1, 1): 1,
    (1, 0): -1,
    (1, -1): -1,
    (0, 1): -1,
    (0, 0): 0,
    (0, -1): -1,
    (-1, 1): -1,
    (-1, 0): -1,
    (-1, -1): -1
}

ternary_opxor_table = {
    (1, 1): -1,
    (1, 0): 1,
    (1, -1): 1,
    (0, 1): 1,
    (0, 0): 0,
    (0, -1): 1,
    (-1, 1): 1,
    (-1, 0): 1,
    (-1, -1): -1
}

def topxor(a, b):
    return ternary_opxor_table[(a, b)]

def tpand(a, b):
    return ternary_pand_table[(a, b)]

def toor(a,b):
    return ternary_oor_table[(a,b)]

def full_op_adder(A, B, Cin):
    sum_op = topxor(A,topxor(B,Cin))
    carry_out_op = toor(tpand(A,B),tpand(Cin,topxor(A,B)))# C'=(A.B)+(C.(A.XOR.B))
    return sum_op, carry_out_op

# for a in [-1,0,1] :
#     for b in [-1,0,1] :
#         for cin in [-1,0,1] :
#             s, c = full_op_adder(a, b, cin)
#             print(f"a={a}, b={b}, cin={cin} -> sum={s}, carry={c}")

def pess_op_ripple(vecs):
    if not vecs:
        return [-1] * 4

    result = vecs[0].copy()
    carry = -1  # neutral

    for vec in vecs[1:]:
        new_result = []
        for a, b in zip(result, vec):
            s, carry = full_op_adder(a, b, carry)
            new_result.append(s)
        result = new_result

    # append the final carry
    if carry != -1:
        result.append(carry)

    return result