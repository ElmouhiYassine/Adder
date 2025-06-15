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

