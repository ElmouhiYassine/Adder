

def kleene_and(a, b):
    """
    Strong Kleene AND for 1 (true), -1 (false), 0 (unknown).
    """
    # false dominates, true only if both true, else unknown
    if a == -1 or b == -1:
        return -1
    if a == 1 and b == 1:
        return 1
    return 0


def kleene_or(a, b):

    # true dominates, false only if both false, else unknown
    if a == 1 or b == 1:
        return 1
    if a == -1 and b == -1:
        return -1
    return 0


def kleene_xor(a, b):

    return min(max(a,b),-min(a,b))


def strong_kleene_full_adder(a, b, cin = -1):

    # sum
    s1 = kleene_xor(a, b)
    sum_bit = kleene_xor(s1, cin)

    # carry-out
    ab = kleene_and(a, b)

    carry_out = kleene_or(ab,kleene_and(cin,s1))

    return sum_bit, carry_out

# Example usage:
if __name__ == "__main__":
    tests = [
        (1, 1, -1),
        (1, 0, -1),
        (0, 0, -1),     # U+U+U
        (-1, -1, -1),   # 0+0+1
        (1, -1, -1),   # 1+0+0
    ]
    for a, b, cin in tests:
        s, c = strong_kleene_full_adder(a, b, cin)
        print(f"a={a}, b={b}, cin={cin} -> sum={s}, carry={c}")
