#%%
import numpy as np

#%%
# Binary case
def half_adder(A, B):
    # Ensure inputs are binary (0 or 1)
    if A not in (0, 1) or B not in (0, 1):
        raise ValueError("Inputs must be 0 or 1")
    
    sum_ = A ^ B      # XOR
    carry = A & B     # AND
    return sum_, carry

# Example usage
# for A in (0, 1):
#     for B in (0, 1):
#         s, c = half_adder(A, B)
#         print(f"A={A}, B={B} → Sum={s}, Carry={c}")
#%%
#Binary case
def full_adder(A, B, Cin):
    # Check that all inputs are binary
    if any(x not in (0, 1) for x in (A, B, Cin)):
        raise ValueError("All inputs must be 0 or 1")
    
    sum_ = A ^ B ^ Cin
    carry_out = (A & B) | (A & Cin) | (B & Cin)
    return sum_, carry_out

# Example usage: full truth table
# print("A B Cin | Sum Carry")
# for A in (0, 1):
#     for B in (0, 1):
#         for Cin in (0, 1):
#             s, c = full_adder(A, B, Cin)
#             print(f"{A} {B}  {Cin}   |  {s}    {c}")
#%%
# Ternary tables
ternary_neg_table ={
    1:0,
    0:1,
    .5:.5  
}

ternary_xor_table = {
    (1, 1): 0,
    (1, .5): .5,
    (1, 0): 1,
    (.5, 1): .5,
    (.5, .5): .5,
    (.5, 0): .5,
    (0, 1): 1,
    (0, .5): .5,
    (0, 0): 0
}

ternary_and_table = {
    (1, 1): 1,
    (1, .5): .5,
    (1, 0): 0,
    (.5, 1): .5,
    (.5, .5): .5,
    (.5, 0): 0,
    (0, 1): 0,
    (0, .5): .5,
    (0, 0): 0
}

ternary_or_table = {
    (1, 1): 1,
    (1, .5): 1,
    (1, 0): 1,
    (.5, 1): 1,
    (.5, .5): .5,
    (.5, 0): .5,
    (0, 1): 1,
    (0, .5): .5,
    (0, 0): 0
}
#%%
#quasi-connectives

ternary_qor_table = {
    (1, 1): 1,
    (1, .5): 1,
    (1, 0): 1,
    (.5, 1): 1,
    (.5, .5): .5,
    (.5, 0): 0,
    (0, 1): 1,
    (0, .5): 0,
    (0, 0): 0
}

ternary_qand_table = {
    (1, 1): 1,
    (1, .5): 1,
    (1, 0): 0,
    (.5, 1): 1,
    (.5, .5): .5,
    (.5, 0): 0,
    (0, 1): 0,
    (0, .5): 0,
    (0, 0): 0
}

ternary_qxor_table = {
    (1, 1): 0,
    (1, .5): 0,
    (1, 0): 1,
    (.5, 1): 0,
    (.5, .5): .5,
    (.5, 0): 0,
    (0, 1): 1,
    (0, .5):1,
    (0, 0): 0
}
#%%
def txor(a, b):
    return ternary_xor_table[(a, b)]

def tand(a, b):
    return ternary_and_table[(a, b)]

def tor(a,b):
    return ternary_or_table[(a,b)]

def tneg(a):
    return ternary_neg_table[(a)]

def tqxor(a, b):
    return ternary_qxor_table[(a, b)]

def tqand(a, b):
    return ternary_qand_table[(a, b)]

def tqor(a,b):
    return ternary_qor_table[(a,b)]
#%%
def sk_hadd(a, b):
    sk_hsum = txor(a,b)
    sk_hcarry = tand(a,b)
    return sk_hsum, sk_hcarry
#%%
# for A in (0,.5, 1):
#     for B in (0,.5, 1):
#         s, c = sk_hadd(A, B)
#         print(f"A={A}, B={B} → skSum={s}, skCarry={c}")
#%%
def full_sk_adder(A, B, Cin):    
    sum_sk = txor(A,txor(B,Cin))
    carry_out_sk = tor(tand(A,B),tand(Cin,txor(A,B)))# C'=(A.B)+(C.(A.XOR.B))
    return sum_sk, carry_out_sk


# Example usage: full truth table
# print("  A   B  Cin | Sum Carry")
# for A in (0,.5, 1):
#     for B in (0,.5, 1):
#         for Cin in (0,.5, 1):
#             s, c = full_sk_adder(A, B, Cin)
#             print(f"{A:>3} {B:>3} {Cin:>4} | {s:>4} {c:>5}")

#%%
def ternary_ripple_adder(A_list, B_list):
    # Pad both lists to the same length
    max_len = max(len(A_list), len(B_list))
    A_list = A_list[::-1] + [0] * (max_len - len(A_list))  # Reverse for LSB-first
    B_list = B_list[::-1] + [0] * (max_len - len(B_list))

    result = []
    carry = 0

    for a, b in zip(A_list, B_list):
        s, carry = full_sk_adder(a, b, carry)
        result.append(s)

    if carry != 0:
        result.append(carry)

    return result[::-1]  # Return to MSB-first order
#%%
# A = [1, 0, .5]   # Represents 4 or 5
# B = [1, 1, .5]   # 6 ou 7
#
# sum_result = ternary_ripple_adder(A, B)
# print("Sum:", sum_result)
#%%
#TODO: tell the system to replace .5 by 0 and calculate the result
# Find a different AND or XOR that could do this, maybe Middle Kleene?
#%%
def full_q_adder(A, B, Cin):    
    sum_q = tqxor(A,tqxor(B,Cin))
    carry_out_q = tqor(tqand(A,B),tqand(Cin,tqxor(A,B)))# C'=(A.B)+(C.(A.XOR.B))
    return sum_q, carry_out_q


# Example usage: full truth table
# print("A B Cin | Sum Carry")
# for A in (0,.5, 1):
#     for B in (0,.5, 1):
#         for Cin in (0,.5, 1):
#             s, c = full_q_adder(A, B, Cin)
#             print(f"{A:>3} {B:>3} {Cin:>4} | {s:>4} {c:>5}")

#%%
def map_quasi_adder(a,b,cin) :
    map ={
        -1 : 0,
        0 : 0.5,
        1: 1,
    }
    mapout ={
        0 : -1,
        0.5 : 0,
        1 : 1,
    }
    a = map[a]
    b = map[b]
    cin = map[cin]
    sum = tqxor(a,tqxor(b,cin))
    carry = tqor(tqand(a,b),tqand(cin,tqxor(a,b)))# C'=(A.B)+(C.(A.XOR.B))

    sum = mapout[sum]
    carry = mapout[carry]
    return sum, carry

# print("A B Cin | Sum Carry")
# for A in (0,-1, 1):
#     for B in (0,-1, 1):
#         for Cin in (0,-1, 1):
#             s, c = map_quasi_adder(A, B, Cin)
#             print(f"{A:>3} {B:>3} {Cin:>4} | {s:>4} {c:>5}")
#%%
def ternary_ripple_qadder(A_list, B_list):
    # Pad both lists to the same length
    max_len = max(len(A_list), len(B_list))
    A_list = A_list[::-1] + [0] * (max_len - len(A_list))  # Reverse for LSB-first
    B_list = B_list[::-1] + [0] * (max_len - len(B_list))

    result = []
    carry = 0

    for a, b in zip(A_list, B_list):
        s, carry = full_q_adder(a, b, carry)
        result.append(s)

    if carry != 0:
        result.append(carry)

    return result[::-1]  # Return to MSB-first order
#%%
# A = [.5, 1, 1]   # Represents min 1
# B = [1, .5, .5]   # min 4
#
# sum_result = ternary_ripple_qadder(A, B)
# print("Sum:", sum_result)
#%%
# A = [1, .5, 1]   # min 5, max 7
# B = [1, .5, .5]   # min 4, max 7: qadd is 13 so closer to 14 than to 9
#
# sum_result = ternary_ripple_qadder(A, B)
# print("Sum:", sum_result)
#%%
# A = [.5, 1, .5]   # min 2, max 7
# B = [.5, 1, .5]   # min 2, max 7: qadd is 15 so closer to 14 than to 4
#
# sum_result = ternary_ripple_qadder(A, B)
# print("Sum:", sum_result)
#%%
# A = [.5, 1, 1]   # min 2, max 7
# B = [.5, 1, 1]   # min 2, max 7: qadd is 13 so closer to 14 than to 9
#
# sum_result = ternary_ripple_adder(A, B)
# print("Sum:", sum_result)
#%%
## Upshot: resolution here is sometimes conservative and sometimes not.