from Adders.strong_kleene_adder import SK_ripple_add
from Adders.balanced_ternary_adder import balanced_ternary_add
from Adders.pesi_op_adder import pess_op_ripple
from Adders.SK_Quasi_adder import ripple_quasi
from Adders.lukasiewicz_adder import lukasiewicz_ripple
from Adders.Ternary_New_Adder import get_Adder
# extend the vector, adding -1
def pad_to_ext(vec_lsb: list[int]) -> list[int]:
    if len(vec_lsb) < U_EXT_WIDTH:
        return vec_lsb + [-1] * (U_EXT_WIDTH - len(vec_lsb))
    return vec_lsb[:U_EXT_WIDTH]

# flip -1 to 1, 1 to -1
def flip_only_pad(vec_lsb: list[int]) -> list[int]:
    vec_lsb = pad_to_ext(vec_lsb)
    flipped = [(-1 if t == +1 else (+1 if t == -1 else 0)) for t in vec_lsb]
    return flipped

def bits_to_int(lsb_vec: list[int]) -> int:
    """Map {-1,+1}→{0,1} and build integer from LSB-first vector."""
    val = 0
    for b, t in enumerate(lsb_vec):
        if t == +1:
            val |= (1 << b)
    return val

def int_to_bits(n: int, width: int) -> list[int]:
    """Integer → {-1,+1} bits (LSB-first), width-limited."""
    return [+1 if (n >> b) & 1 else -1 for b in range(width)]
def flip_only_extend_LSBfirst(u_lsb: list, K1: int):
    """
    One's-complement flip for uncertain-binary trits: +1<->-1, 0 stays 0.
    Input:  K1-digit vector LSB-first.
    Output: (K1+1)-digit vector LSB-first (prepends a -1 baseline on MSB side, then flips all).
    """
    msb = list(reversed(u_lsb))
    msb = [-1] + msb  # extend by a leading -1 baseline (MSB side)
    flipped_msb = []
    for b in msb:
        if b == +1:
            flipped_msb.append(-1)
        elif b == -1:
            flipped_msb.append(+1)
        else:
            flipped_msb.append(0)
    return list(reversed(flipped_msb))  # back to LSB-first, length K1+1

# small helpers for U certain-path math
def bits_to_int(lsb_vec: list[int]) -> int:
    """Map {-1,+1}→{0,1} and build integer from LSB-first vector."""
    val = 0
    for b, t in enumerate(lsb_vec):
        if t == +1:
            val |= (1 << b)
    return val

def int_to_bits(n: int, width: int) -> list[int]:
    """Integer → {-1,+1} bits (LSB-first), width-limited."""
    return [+1 if (n >> b) & 1 else -1 for b in range(width)]
def int_to_binary(n: int, width: int):
    """Encode integer n as {+1,-1} (LSB-first) for uncertain-binary adder."""
    out = []
    for i in range(width):
        out.append(+1 if (n & (1 << i)) else -1)
    return out

U_EXT_WIDTH = 10
def test(taps):
    terms = []
    neg_count = 0
    for kval, uvec in taps:
        if kval == +1:
            term = pad_to_ext(uvec)
        else:
            term = flip_only_pad(uvec)
            neg_count += 1
        terms.append(term)
    print(terms)
    acc = terms[0]
    for t in terms[1:]:
        acc = SK_ripple_add([acc, t])
        acc = pad_to_ext(acc)

    if neg_count:
        corr = int_to_binary(neg_count, width=U_EXT_WIDTH)
        print(corr)
        acc = SK_ripple_add([acc, corr])
        acc = pad_to_ext(acc)

    return acc


def twos_complement_(uvec_lsb: list[int], U_EXT_WIDTH: int) -> list[int]:

    # 1. Pad the input vector and flip LSB/MSB order for easier traversal
    full_vector = uvec_lsb + [-1] * (U_EXT_WIDTH - len(uvec_lsb))

    # 2. Iterate and apply the rule
    result = []
    found_one = False

    for i in range(len(full_vector)):
        trit = full_vector[i]

        if not found_one:
            # Step 1 & 2: Copy until the first +1
            if trit == +1:
                result.append(+1)
                found_one = True
            elif trit == -1:
                result.append(-1)
            elif trit == 0:
                result.append(0)
        else:
            # Step 3: Invert all remaining bits
            if trit == +1:
                result.append(-1)
            elif trit == -1:
                result.append(+1)
            elif trit == 0:
                # Uncertainty propagates through inversion
                result.append(0)

    return result

print(twos_complement_([0,1,1,-1,1], 10))