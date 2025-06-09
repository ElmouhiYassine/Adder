def Normal_adder(a, b, cin):

    sum1, carry1 = ternary_half_adder(a, b)

    final_sum, carry2 = ternary_half_adder(sum1, cin)

    final_carry = 1 if carry1 == 1 or carry2 == 1 else -1

    return final_sum, final_carry

def ternary_half_adder(a, b):
    # XOR
    sum_val = 1 if a * b == -1 else -1
    # AND
    carry_val = 1 if a == 1 and b == 1 else -1
    return sum_val, carry_val

# Test
inputs = [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
          (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]

# for a, b, cin in inputs:
#     s, c = Normal_adder(a, b, cin)
#     print(f"a={a}, b={b}, cin={cin} => sum={s}, carry={c}")
