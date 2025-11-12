import numpy as np

def neg_pess(x: int) -> int:
    if x == 1:
        return -1
    if x == -1:
        return 1
    if x == 0:
        return 0


def pessimistic_and(a: int, b: int) -> int:
    """
       AND |  -1    0     +1
       ----------------------
        -1  |  -1    -1    -1
         0  |  -1     0    -1
        +1  |  -1    -1    +1
    """
    if a == 0 and b == 0:
        return 0
    if a == 1 and b == 1:
        return 1
    return -1


def pessimistic_or(a: int, b: int) -> int:
    """
       OR  |  -1    0     +1
       ----------------------
        -1  |  -1    +1    +1
         0  |  +1     0    +1
        +1  |  +1    +1    +1
    """
    if a == 0 and b == 0:
        return 0
    if a == -1 and b == -1:
        return -1
    return 1



def pessimistic_xor(a: int, b: int) -> int:
    """
       XOR |  -1    0     +1
       ----------------------
        -1  |  -1    +1    +1
         0  |  +1     0    +1
        +1  |  +1    +1    -1
    """

    xor_table = {
            (-1, -1): -1,
            (-1, 0): +1,
            (-1, +1): +1,
            (0, -1): +1,
            (0, 0): 0,
            (0, +1): +1,
            (+1, -1): +1,
            (+1, 0): +1,
            (+1, +1): -1
        }
    return xor_table[(a, b)]


def pessimistic_full_adder(a: int, b: int, cin: int = -1) -> (int, int):

    s1 = pessimistic_xor(a, b)
    S  = pessimistic_xor(s1, cin)

    c_ab = pessimistic_and(a, b)
    c_sc = pessimistic_and(cin, s1)
    Cout = pessimistic_or(c_ab, c_sc)

    return S, Cout


if __name__ == "__main__":
    # Test all 3×3×3 = 27 input combinations:
    values = [-1, 0, 1]
    print(" a   b  cin |  sum  carry")
    print("-------------------------")
    for a in values:
        for b in values:
            for cin in values:
                s, c = pessimistic_full_adder(a, b, cin)
                print(f"{a:>2}  {b:>2}  {cin:>2} |   {s:>2}    {c:>2}")
