ternary_tor_table = {
    (1, 1): 1,
    (1, 0): 1,
    (1, -1): 1,
    (0, 1): 1,
    (0, 0): 0,
    (0, -1): -1,
    (-1, 1): 1,
    (-1, 0): -1,
    (-1, -1): -1
}

ternary_tand_table = {
    (1, 1): 1,
    (1, 0): 0,
    (1, -1): 0,
    (0, 1): 0,
    (0, 0): 0,
    (0, -1): 0,
    (-1, 1): 0,
    (-1, 0): 0,
    (-1, -1): -1
}

ternary_tneg_table = {
    1: -1,
    -1: 1,
    0: 0
}


def tor(a, b):
    return ternary_tor_table[(a, b)]


def tand(a, b):
    return ternary_tand_table[(a, b)]


def tneg(a):
    return ternary_tneg_table[(a)]


def sum_half(a, b):  # first interpretation of diagram
    if (a, b) == (0, 0):
        return (0, 0)
    else:
        return tor(tor(a, b), tneg(tand(a, b)))


values = [-1, 0, 1]

for a in values:
    for b in values:
        result = tor(a, b)
        print(f"sum_half({a}, {b}) = {result}")

        ## does not reflect diagram!