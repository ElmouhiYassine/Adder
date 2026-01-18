import numpy as np
import math
from typing import Callable, Tuple, Any

from Adders.sobocinski_adder import sobocinski_ripple
# Import required helpers
from HTLConvolution.helpers import (
    fit_width,
    int_to_balanced_ternary,
)


# ========================================================
#                     U PASS
# ========================================================

def U_pass(U: np.ndarray,
           kernel: np.ndarray,
           ripple_add: Callable,
           U_EXT_WIDTH: int = 10,
           carry_policy: str = "definite"
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fixed-width uncertainty pass over U (uncertain binary).
    Produces:
      - U_trunc: low K1 bits (LSB-first)
      - carry_vals: signed carry-injected values
      - carry_mask: certainty mask
    """

    H, W, K1 = U.shape
    k = kernel.shape[0]
    out_H, out_W = H - k + 1, W - k + 1

    U_trunc = np.full((out_H, out_W, K1), -1, dtype=int)
    carry_vals = np.zeros((out_H, out_W), dtype=int)
    carry_mask = np.full((out_H, out_W), -1, dtype=int)

    # --------- local helpers (kept here because U_pass uses them deeply) ---------

    def pad_to_ext(vec_lsb: list[int]) -> list[int]:
        if len(vec_lsb) < U_EXT_WIDTH:
            return vec_lsb + [-1] * (U_EXT_WIDTH - len(vec_lsb))
        return vec_lsb[:U_EXT_WIDTH]

    def bits_to_int(lsb_vec: list[int]) -> int:
        val = 0
        for b, t in enumerate(lsb_vec):
            if t == +1:
                val |= (1 << b)
        return val

    def int_to_bits(n: int, width: int) -> list[int]:
        return [+1 if (n >> b) & 1 else -1 for b in range(width)]

    def twos_complement_(vec, width, adder_func):
        # -------------------------------------------------
        # 1) Pad to full width
        # -------------------------------------------------
        x = vec[:] + [-1] * (width - len(vec))  # -1 = definite 0

        # -------------------------------------------------
        # 2) Bitwise flip
        #    1 -> -1
        #   -1 ->  1
        #    0 ->  0   (uncertainty preserved)
        # -------------------------------------------------
        flipped = []
        for t in x:
            if t == 1:
                flipped.append(-1)
            elif t == -1:
                flipped.append(1)
            else:
                flipped.append(0)

        # -------------------------------------------------
        # 3) Add +1 using the uncertain adder
        # -------------------------------------------------
        one = [1] + [-1] * (width - 1)  # +1 in LSB-first ternary
        return adder_func([one,flipped])
        # result = []
        # for a, b in zip(flipped, one):
        #     s, carry = adder_func(a, b, carry)
        #     result.append(s)
        #
        # # Ignore overflow beyond width
        # return result

    def resolve_carry(c_segment: list[int], start_bit: int) -> tuple[int, int]:
        """Resolve carries for ± uncertainty policies."""
        num_bits = len(c_segment)
        sign_bit_index = num_bits - 1
        has_zero = any(t == 0 for t in c_segment)

        # --- definite ---
        if not has_zero:
            val = 0
            for i, trit in enumerate(c_segment):
                bit_pos = start_bit + i
                if trit == +1:
                    if i == sign_bit_index:
                        val -= (1 << bit_pos)
                    else:
                        val += (1 << bit_pos)
            return val, 1

        # --- uncertain ---
        if carry_policy == "definite":
            return 0, 1

        elif carry_policy == "max":
            val = 0
            for i, trit in enumerate(c_segment):
                bit_pos = start_bit + i
                is_sign = (i == sign_bit_index)
                eff = trit
                if trit == 0:
                    eff = -1 if is_sign else +1
                if eff == +1:
                    if is_sign:
                        val -= (1 << bit_pos)
                    else:
                        val += (1 << bit_pos)
            return val, 1

        elif carry_policy == "min":
            val = 0
            for i, trit in enumerate(c_segment):
                bit_pos = start_bit + i
                is_sign = (i == sign_bit_index)
                eff = trit
                if trit == 0:
                    eff = +1 if is_sign else -1
                if eff == +1:
                    if is_sign:
                        val -= (1 << bit_pos)
                    else:
                        val += (1 << bit_pos)
            return val, 1

        elif carry_policy == "center":
            valf = 0.0
            for i, trit in enumerate(c_segment):
                bit_pos = start_bit + i
                is_sign = (i == sign_bit_index)
                p1 = 1.0 if trit == +1 else (0.0 if trit == -1 else 0.5)
                if is_sign:
                    valf -= p1 * (1 << bit_pos)
                else:
                    valf += p1 * (1 << bit_pos)
            return int(round(valf)), 1

        return 0, 1

    # ======================================================
    #                 MAIN LOOP
    # ======================================================

    for i in range(out_H):
        for j in range(out_W):
            taps = []
            all_certain = True

            for dx in range(k):
                for dy in range(k):
                    kval = kernel[dx, dy]
                    if kval == 0:
                        continue

                    uvec = U[i + dx, j + dy, :].tolist()
                    if any(t == 0 for t in uvec):
                        all_certain = False
                    taps.append((kval, uvec))

            if not taps:
                continue

            if all_certain:
                base = (1 << K1)
                acc_sum = 0
                for kval, u in taps:
                    term = bits_to_int(u)
                    acc_sum += term if kval == +1 else -term

                carry_units = math.floor(acc_sum / base)
                low = acc_sum - carry_units * base

                U_trunc[i, j, :] = np.array(int_to_bits(low, K1), dtype=int)
                carry_vals[i, j] = carry_units * base
                carry_mask[i, j] = (+1 if carry_units != 0 else -1)

            else:
                terms = []
                for kval, u in taps:
                    if kval == +1:
                        t = pad_to_ext(u)
                    else:
                        t = twos_complement_(u, U_EXT_WIDTH,ripple_add)
                    terms.append(t)

                acc = terms[0]
                for t in terms[1:]:
                    acc = ripple_add([acc, t])
                    acc = pad_to_ext(acc)

                U_trunc[i, j, :] = acc[:K1]
                cseg = acc[K1:U_EXT_WIDTH]
                val, mask = resolve_carry(cseg, K1)
                carry_vals[i, j] = val
                carry_mask[i, j] = mask

    return U_trunc, carry_vals, carry_mask


# ========================================================
#                     V PASS
# ========================================================

def V_pass(V: np.ndarray,
           kernel: np.ndarray,
           balanced_ternary_add: Callable):
    """
    Balanced-ternary convolution on V (upper K2 trits).
    """
    H, W, K2 = V.shape
    k = kernel.shape[0]
    out_H, out_W = H - k + 1, W - k + 1
    V_out = np.zeros((out_H, out_W, K2), dtype=int)

    for i in range(out_H):
        for j in range(out_W):
            acc = None

            for dx in range(k):
                for dy in range(k):
                    kval = kernel[dx, dy]
                    if kval == 0:
                        continue

                    vec = V[i + dx, j + dy, :].tolist()
                    if kval == -1:
                        vec = [-t for t in vec]

                    if acc is None:
                        acc = vec
                    else:
                        L = max(len(acc), len(vec)) + 1
                        acc = balanced_ternary_add(
                            fit_width(acc, L, 0),
                            fit_width(vec, L, 0)
                        )

            if acc is not None:
                V_out[i, j, :] = np.array(fit_width(acc, K2, 0), dtype=int)

    return V_out


# ========================================================
#                   FULL CONVOLUTION
# ========================================================

def convolution(X: np.ndarray,
                kernel: np.ndarray,
                ripple_add: Callable,
                balanced_ternary_add: Callable,
                K1: int = 5,
                K2: int = 5,
                return_info: bool = False):
    """
    Full (K1+K2)-trit hybrid convolution.
    """
    H, W, _ = X.shape
    U = X[:, :, :K1]
    V = X[:, :, K1:]

    U_trunc, carry_vals, carry_mask = U_pass(
        U, kernel, ripple_add, carry_policy="max"
    )

    V_pre = V_pass(V, kernel, balanced_ternary_add)
    H_out, W_out, _ = V_pre.shape

    V_final = V_pre.copy()

    # Inject signed carry into V
    for i in range(H_out):
        for j in range(W_out):
            amt = int(carry_vals[i, j])
            if amt != 0 and carry_mask[i, j] != 0:
                off = int_to_balanced_ternary(abs(amt), K2)
                if amt < 0:
                    off = [-d for d in off]

                curr = fit_width(V_final[i, j, :].tolist(), K2, 0)
                tmp = balanced_ternary_add(curr, off)
                V_final[i, j, :] = np.array(fit_width(tmp, K2, 0), dtype=int)

    Y = np.concatenate([U_trunc, V_final], axis=2)
    if return_info:
        return Y, {
            "U_trunc": U_trunc,
            "carry_vals": carry_vals,
            "carry_mask": carry_mask,
            "V_pre": V_pre,
        }
    return Y


# def twos_complement_(uvec_lsb: list[int], width: int) -> list[int]:
#     """LSB-first two's complement for uncertain-binary trits."""
#     full_vec = uvec_lsb + [-1] * (width - len(uvec_lsb))
#
#     out = []
#     found_one = False
#
#     for t in full_vec:
#         if not found_one:
#             out.append(+1 if t == +1 else -1 if t == -1 else 0)
#             if t == +1:
#                 found_one = True
#         else:
#             if t == +1:
#                 out.append(-1)
#             elif t == -1:
#                 out.append(+1)
#             else:
#                 out.append(0)
#
#     return out
#
# def pad_to_ext(vec_lsb: list[int]) -> list[int]:
#     if len(vec_lsb) < 10:
#         return vec_lsb + [-1] * (10 - len(vec_lsb))
#     return vec_lsb[:10]
#
# acc = sobocinski_ripple([[1,1,1,1,1,1], twos_complement_([-1,1,1,1,1,1],10)])
# acc = pad_to_ext(acc)
#
# print(acc)

# Execute with "center" policy and vec [0,1,1,1,1,1]

# def resolve_carry(c_segment: list[int], start_bit: int, carry_policy: str) -> tuple[int, int]:
#     """Resolve carries for ± uncertainty policies."""
#     num_bits = len(c_segment)
#     sign_bit_index = num_bits - 1
#     has_zero = any(t == 0 for t in c_segment)
#
#     # --- definite ---
#     if not has_zero:
#         val = 0
#         for i, trit in enumerate(c_segment):
#             bit_pos = start_bit + i
#             if trit == +1:
#                 if i == sign_bit_index:
#                     val -= (1 << bit_pos)
#                 else:
#                     val += (1 << bit_pos)
#         return val, 1
#
#     # --- uncertain ---
#     if carry_policy == "definite":
#         return 0, 1
#
#     elif carry_policy == "max":
#         val = 0
#         for i, trit in enumerate(c_segment):
#             bit_pos = start_bit + i
#             is_sign = (i == sign_bit_index)
#             eff = trit
#             if trit == 0:
#                 eff = -1 if is_sign else +1
#             if eff == +1:
#                 if is_sign:
#                     val -= (1 << bit_pos)
#                 else:
#                     val += (1 << bit_pos)
#         return val, 1
#
#     elif carry_policy == "min":
#         val = 0
#         for i, trit in enumerate(c_segment):
#             bit_pos = start_bit + i
#             is_sign = (i == sign_bit_index)
#             eff = trit
#             if trit == 0:
#                 eff = +1 if is_sign else -1
#             if eff == +1:
#                 if is_sign:
#                     val -= (1 << bit_pos)
#                 else:
#                     val += (1 << bit_pos)
#         return val, 1
#
#     elif carry_policy == "center":
#         valf = 0.0
#         for i, trit in enumerate(c_segment):
#             bit_pos = start_bit + i
#             is_sign = (i == sign_bit_index)
#             p1 = 1.0 if trit == +1 else (0.0 if trit == -1 else 0.5)
#             if is_sign:
#                 valf -= p1 * (1 << bit_pos)
#             else:
#                 valf += p1 * (1 << bit_pos)
#         return int(round(valf)), 1
#
#     return 0, 1
# vec = [0, 1, 1, 1, 1, 1]
# result = resolve_carry(vec, start_bit=4, carry_policy="max")
# print(f"Input vector: {vec}")
# print(f"Carry policy: center")
# print(f"Result: {result}")
# print(f"\nDetailed breakdown:")
# print(f"Value: {result[0]}, Flags: {result[1]}")