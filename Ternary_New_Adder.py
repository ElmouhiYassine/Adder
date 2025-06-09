import json

def str_to_tuple(key):
    # Convert string keys like '0,1' to tuple (0, 1)
    return tuple(map(int, key.split(',')))

def load_gates():
    with open('ternary_gates.json', 'r') as f:
        gates = json.load(f)
    # Convert string keys to tuples for all gate pairs
    for gate_pair in gates:
        gate_pair['and_gate'] = {str_to_tuple(k): v for k, v in gate_pair['and_gate'].items()}
        gate_pair['or_gate'] = {str_to_tuple(k): v for k, v in gate_pair['or_gate'].items()}
    return gates

def fetch_gates(n):
    gates = load_gates()
    if n < 0 or n >= len(gates):
        raise IndexError("Gate index out of range")
    return gates[n]['and_gate'], gates[n]['or_gate']

def neg(x):
    return -x if x != 0 else 0

def make_and(and_gate):
    def and_func(a, b):
        return and_gate[(a, b)]
    return and_func

def make_or(or_gate):
    def or_func(a, b):
        return or_gate[(a, b)]
    return or_func

def make_xor(and_gate, or_gate):
    and_func = make_and(and_gate)
    or_func = make_or(or_gate)
    def xor_func(a, b):
        a_and_b = and_func(a, b)
        not_a_and_b = neg(a_and_b)
        a_or_b = or_func(a, b)
        return and_func(a_or_b, not_a_and_b)
    return xor_func

def make_full_adder(and_gate, or_gate):
    and_func = make_and(and_gate)
    or_func = make_or(or_gate)
    xor_func = make_xor(and_gate, or_gate)
    def full_adder(a, b, carry_in):
        # Compute sum: (a XOR b) XOR carry_in
        xor_ab = xor_func(a, b)
        sum_out = xor_func(xor_ab, carry_in)
        # Compute carry: (a AND b) OR ((a XOR b) AND carry_in)
        and_ab = and_func(a, b)
        xor_ab_and_carry = and_func(xor_ab, carry_in)
        carry_out = or_func(and_ab, xor_ab_and_carry)
        return sum_out, carry_out
    return full_adder

def get_Adder(key):
    and_gate, or_gate = fetch_gates(key)
    full_adder = make_full_adder(and_gate, or_gate)
    return full_adder

# Example usage
if __name__ == "__main__":
    # Fetch the first gate pair (n=0)
    and_gate, or_gate = fetch_gates(0)
    # Create the full adder function
    full_adder = make_full_adder(and_gate, or_gate)
    # Test with sample inputs
    test_cases = [
        (1, 1, 0),
        (-1, -1, 0),
        (1, -1, 0),
        (0, 1, 0),
    ]
    for a, b, carry_in in test_cases:
        sum_out, carry_out = full_adder(a, b, carry_in)
        print(f"Inputs: a={a}, b={b}, carry_in={carry_in} -> Sum: {sum_out}, Carry: {carry_out}")