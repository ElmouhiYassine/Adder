
# Cyclical addition for ternary with carry-in
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
    """
    Big-endian cyclical ternary adder.

    Params:
    - A, B: Lists of equal length (MSD at index 0)
    - logic_system: contains 'add' and 'digits'

    Returns:
    - S: Sum list (big-endian)
    - C: Final carry
    """
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
# Combine function: adds final carry to front of result
def combine_sum_and_carry(sum_list, carry):
    """
    Add final carry to the most significant side (prepend it).
    """
    return [carry] + sum_list


# -------------------------------------------------------
# TESTING
def test_big_endian_cyclical():
    A = [1, 1]
    B = [1, 1]
    S, C = cyclical_adder(A, B, cyclical_ternary_logic)
    result = combine_sum_and_carry(S, C)
    print(f"{A} + {B} => Sum = {S}, Carry = {C}, Result = {result}")

    A2 = [-1, -1]
    B2 = [-1, -1]
    S2, C2 = cyclical_adder(A2, B2, cyclical_ternary_logic)
    result2 = combine_sum_and_carry(S2, C2)
    print(f"{A2} + {B2} => Sum = {S2}, Carry = {C2}, Result = {result2}")

    A3 = [1, -1, 0]
    B3 = [0, 1, -1]
    S3, C3 = cyclical_adder(A3, B3, cyclical_ternary_logic)
    result3 = combine_sum_and_carry(S3, C3)
    print(f"{A3} + {B3} => Sum = {S3}, Carry = {C3}, Result = {result3}")


if __name__ == "__main__":
    test_big_endian_cyclical()
