# Ternary XOR lookup table
ternary_xor_table = {
    (1, 1): -1,
    (1, 0): 0,
    (1, -1): 1,
    (0, 1): 0,
    (0, 0): 0,
    (0, -1): 0,
    (-1, 1): 1,
    (-1, 0): 0,
    (-1, -1): -1
}

# logic systems
binary_logic = {
    'and': lambda a, b: a & b,
    'or': lambda a, b: a | b,
    'xor': lambda a, b: a ^ b,
    'digits': {0, 1}
}

ternary_logic = {
    'and': lambda a, b: min(a, b),
    'or': lambda a, b: max(a, b),
    'xor': lambda a, b: ternary_xor_table[(a, b)],
    'digits': {-1, 0, 1}
}

def adder(A, B, logic_system):
    """
    Parameters:
    - A: List of N digits
    - B: List of N digits (same length as A)
    - logic_system: Dictionary with 'and', 'or', 'xor' functions and 'digits' set

    Returns:
    - S: List of N sum digits
    - C: Final carry after N digits
    """

    if len(A) != len(B):
        raise ValueError("A and B must have the same length")

    N = len(A)
    # Validate inputs
    allowed_digits = logic_system['digits']
    for a, b in zip(A, B):
        if a not in allowed_digits or b not in allowed_digits:
            raise ValueError(f"Inputs must be in {allowed_digits}")

    S = [0] * N
    C = 0  # Initial carry

    for i in range(N):
        # sum: S[i] = A[i] XOR B[i] XOR C[i-1]
        xor_ab = logic_system['xor'](A[i], B[i])
        S[i] = logic_system['xor'](xor_ab, C)

        #carry: C[i] = (A[i] AND B[i]) OR (C[i-1] AND (A[i] XOR B[i]))
        C = logic_system['or'](
            logic_system['and'](A[i], B[i]),
            logic_system['and'](C, xor_ab)
        )

    return S, C



# Testing the implementation
def test_adder():
    # Binary test: 5 (101) + 6 (110) = 11 (1011)
    # Little-endian format : the leftmost is the least significant
    A_bin = [1, 0, 1]  # 5
    B_bin = [0, 1, 1]  # 6
    S_bin, C_bin = adder(A_bin, B_bin, binary_logic)
    print(f"Binary: {A_bin} + {B_bin} = {S_bin}, Carry = {C_bin}")
    # Expected: S = [1, 1, 0], C = 1 (11 in binary)

    # Ternary test with known values: [1, 1] + [-1, 1]
    A_tern = [1, 1]
    B_tern = [1, 1]
    S_tern, C_tern = adder(A_tern, B_tern, ternary_logic)
    print(f"Ternary: {A_tern} + {B_tern} = {S_tern}, Carry = {C_tern}")

    # Ternary test with unknown (U = 0)
    A_tern_u = [1, 0]
    B_tern_u = [-1, 1]
    S_tern_u, C_tern_u = adder(A_tern_u, B_tern_u, ternary_logic)
    print(f"Ternary with U: {A_tern_u} + {B_tern_u} = {S_tern_u}, Carry = {C_tern_u}")


if __name__ == "__main__":
    test_adder()


## part 2 : imagine for instance that is could also be based on a condition C ,
# A[i] XOR B [i] = 1 if condition C1, else 0 if condition C2 otherwise U
# -------------------------------------------------------------------------
def conditional_xor(a, b):
    if a == 1 and b == -1:  # Condition C1
        return 1
    elif a == -1 and b == 1:  # Condition C2
        return 0
    else:  # Otherwise
        return 0  # U

conditional_logic = {
    'and': lambda a, b: min(a, b),
    'or': lambda a, b: max(a, b),
    'xor': conditional_xor,
    'digits': {-1, 0, 1}
}
#--------------------------------------------------------------
