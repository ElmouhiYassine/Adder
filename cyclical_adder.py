
# Cyclical addition
def cyclical_add(a, b, c):
    total = a + b + c

    if total > 1:
        return total - 3, 1
    elif total < -1:
        return total + 3, -1
    else:
        return total, 0


# Logic system
cyclical_ternary_logic = {
    'add': cyclical_add,
    'digits': {-1, 0, 1}
}


def cyclical_adder(A, B, logic_system = cyclical_ternary_logic ):

    if len(A) != len(B):
        raise ValueError("A and B must be of same length")

    N = len(A)
    allowed = logic_system['digits']
    for a, b in zip(A, B):
        if a not in allowed or b not in allowed:
            raise ValueError(f"Invalid digit. Allowed: {allowed}")

    S = [0] * N
    carry = 0

    for i in reversed(range(N)):  # Iterate from least significant (right) to most significant (left)
        S[i], carry = logic_system['add'](A[i], B[i], carry)

    return S, carry


# -------------------------------------------------------
# adds final carry to front of result
def combine_sum_and_carry(sum_list, carry):
    return [carry] + sum_list

# -------------------------------------------------------
# TESTING
def test():

    A3 = [1, -1, 0]
    B3 = [0, 1, -1]
    S3, C3 = cyclical_adder(A3, B3, cyclical_ternary_logic)
    result3 = combine_sum_and_carry(S3, C3)
    print(f"{A3} + {B3} => Result = {result3}")


if __name__ == "__main__":
    test()
