from Adders.strong_kleene import SK_ripple_add
from Adders.Balanced_Ternary_Adder import balanced_ternary_add

# ---------- small helpers ----------
def fit_width(v, w, pad=0):
    return v[:w] if len(v) >= w else v + [pad] * (w - len(v))

def int_to_bt(n, L):
    if n == 0: v = [0]
    else:
        v, x = [], n
        while x != 0:
            r = x % 3
            if r == 0: v.append(0);  x //= 3
            elif r == 1: v.append(+1); x = (x - 1) // 3
            else: v.append(-1); x = (x + 1) // 3
    return fit_width(v, L, 0)

def int_to_bits_pm1(n, width):
    return [+1 if (n >> b) & 1 else -1 for b in range(width)]

def bits_pm1_to_int(bits):
    val = 0
    for i, t in enumerate(bits):
        if t == +1: val |= (1 << i)
    return val

def decode_bt(vec_bt):
    return sum(int(d) * (3 ** i) for i, d in enumerate(vec_bt))

def show(name, v):
    print(f"{name}: [{', '.join(str(t) for t in v)}]")

# ---------- representations (NO uncertainty) ----------
def build_full_certain_vec(x, U_EXT_WIDTH):
    # bits 0..U_EXT_WIDTH-2 from x; last bit = baseline -1 (guard)
    out = [(+1 if ((x >> b) & 1) else -1) for b in range(U_EXT_WIDTH-1)]
    out.append(-1)
    return out

def build_U_only_vec_certain(x, K1, U_EXT_WIDTH):
    # low K1 bits from x; then pad with -1 up to U_EXT_WIDTH-1; last is -1
    out = [(+1 if ((x >> b) & 1) else -1) for b in range(K1)]
    out.extend([-1] * ((U_EXT_WIDTH-1) - len(out)))
    out.append(-1)
    return out

def extract_carry_window(acc_vec, K1, U_EXT_WIDTH):
    # carry lives in bits [K1 .. U_EXT_WIDTH-2] for the split-injection scheme
    seg = acc_vec[K1:U_EXT_WIDTH-1]
    if any(t == 0 for t in seg):
        return 0, 0  # would mean "uncertain"; not possible here
    val = 0
    for pos, t in enumerate(seg, start=K1):
        if t == +1: val |= (1 << pos)
    return val, (+1 if val > 0 else -1)

def test_split_sum_no_uncertainty(A=173, B=89, K1=4, U_EXT_WIDTH=9, K2=10):
    print(f"\nA={A}, B={B}, K1={K1}, U_EXT_WIDTH={U_EXT_WIDTH}, K2={K2} (NO uncertainty)")

    # Split representation: low K1 into U; upper-aligned ((x>>K1)<<K1) into V (BT)
    U_A = build_U_only_vec_certain(A, K1, U_EXT_WIDTH)
    U_B = build_U_only_vec_certain(B, K1, U_EXT_WIDTH)
    V_A = int_to_bt(((A >> K1) << K1), K2)
    V_B = int_to_bt(((B >> K1) << K1), K2)

    print("\n[Split representation (certain)]")
    show("A.U", U_A)
    show("A.V (BT)", V_A)
    show("B.U", U_B)
    show("B.V (BT)", V_B)

    # Monolithic SK add over full 9-trit vectors (still certain: no 0’s)
    A_full = build_full_certain_vec(A, U_EXT_WIDTH)
    B_full = build_full_certain_vec(B, U_EXT_WIDTH)
    mono   = SK_ripple_add([A_full, B_full])
    mono9  = fit_width(mono, U_EXT_WIDTH, -1)

    # Decode the FULL 9 bits (do not drop the overflow/guard bit here)
    mono_total = bits_pm1_to_int(mono9)   # include bit 8 if it flipped to +1
    mono_low   = mono9[:K1]
    mono_carry_win_val, mono_carry_win_mask = extract_carry_window(mono9, K1, U_EXT_WIDTH)

    print("\n[SK sum (full certain vectors)]")
    show("A_full", A_full)
    show("B_full", B_full)
    show("SK sum (first 9 trits)", mono9)
    show("low K1", mono_low)
    print(f"carry window [K1..{U_EXT_WIDTH-2}] (val, mask): {mono_carry_win_val} {mono_carry_win_mask}")
    print(f"decoded integer (FULL 9 bits) = {mono_total}")

    # Split: U-only SK + carry injection into V (BT)
    U_sum = SK_ripple_add([U_A, U_B]); U_sum = fit_width(U_sum, U_EXT_WIDTH, -1)
    split_low = U_sum[:K1]
    carry_val, carry_mask = extract_carry_window(U_sum, K1, U_EXT_WIDTH)

    V_sum = balanced_ternary_add(V_A, V_B); V_sum = fit_width(V_sum, K2, 0)
    if carry_mask == +1 and carry_val > 0:
        V_sum = balanced_ternary_add(V_sum, int_to_bt(carry_val, K2))
        V_sum = fit_width(V_sum, K2, 0)

    split_total = bits_pm1_to_int(split_low) + decode_bt(V_sum)

    print("\n[Split sum (U-only + carry → V)]")
    show("U_sum (first 9 trits)", U_sum)
    show("split low K1", split_low)
    print(f"carry window (val, mask): {carry_val} {carry_mask}")
    show("V_sum after injection (BT)", V_sum)
    print(f"decoded integer (split)      = {split_total}")

    # Ground truth
    true_sum = A + B
    print(f"\nGround truth A+B             = {true_sum}")

    # Checks
    assert mono_total == true_sum, "FULL decode should equal ground truth."
    assert split_total == true_sum, "Split decode should equal ground truth."
    assert split_low == mono_low,   "Low K1 bits should match (same low bits)."

    print("\n✅ Noise-free check passed: monolithic == split == ground truth.")

if __name__ == "__main__":
    test_split_sum_no_uncertainty(A=204, B=32, K1=4, U_EXT_WIDTH=9, K2=10)
