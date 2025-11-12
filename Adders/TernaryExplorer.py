import itertools
import json
from typing import Dict, Tuple, List


class TernaryLogicGenerator:
    def __init__(self):
        self.values = [-1, 0, 1]

    def neg(self, x):
        return -x if x != 0 else 0

    def check_de_morgan(self, and_gate, or_gate):
        """Check De Morgan's Law: NEG(A OR B) = (NEG A AND NEG B)"""
        for a in self.values:
            for b in self.values:
                if self.neg(or_gate[(a, b)]) != and_gate[(self.neg(a), self.neg(b))]:
                    return False
        return True

    def check_monotonicity(self, and_gate, or_gate):
        """Check monotonicity: if A ≤ B, then (A AND C) ≤ (B AND C) and (A OR C) ≤ (B OR C)"""
        # Order: -1 ≤ 0 ≤ 1
        order = {-1: 0, 0: 1, 1: 2}

        for a in self.values:
            for b in self.values:
                if order[a] <= order[b]:  # a ≤ b
                    for c in self.values:
                        # Check AND monotonicity
                        if order[and_gate[(a, c)]] > order[and_gate[(b, c)]]:
                            return False

                        # Check OR monotonicity
                        if order[or_gate[(a, c)]] > order[or_gate[(b, c)]]:
                            return False
        return True


    def is_valid_gates(self, and_gate, or_gate):
        """Check essential conditions only"""

        # 1. De Morgan's Law (fundamental)
        if not self.check_de_morgan(and_gate, or_gate):
            return False

        # 2. Monotonicity (ensures predictable uncertainty behavior)
        if not self.check_monotonicity(and_gate, or_gate):
            return False

        return True

    def generate_valid_gates(self):
        # Standard binary logic for known values
        and_known = {(1, 1): 1, (-1, -1): -1, (1, -1): -1, (-1, 1): -1}
        or_known = {(1, 1): 1, (-1, -1): -1, (1, -1): 1, (-1, 1): 1}

        # Only unknown combinations to explore
        unique_pairs = [(0, 0), (0, 1), (0, -1)]
        print(f"Exploring 3^3 = 27 possibilities per gate, total 729 combinations")
        print("Checking: De Morgan's Law + Monotonicity + Commutativity")

        valid_gates = set()
        total_checked = 0

        for and_vals in itertools.product(self.values, repeat=3):
            for or_vals in itertools.product(self.values, repeat=3):
                total_checked += 1

                # Build complete gates with commutativity
                and_gate = and_known.copy()
                or_gate = or_known.copy()

                for i, (a, b) in enumerate(unique_pairs):
                    and_gate[(a, b)] = and_vals[i]
                    and_gate[(b, a)] = and_vals[i]  # Commutativity
                    or_gate[(a, b)] = or_vals[i]
                    or_gate[(b, a)] = or_vals[i]  # Commutativity

                if self.is_valid_gates(and_gate, or_gate):
                    gate_tuple = (frozenset(and_gate.items()), frozenset(or_gate.items()))
                    valid_gates.add(gate_tuple)

                if total_checked % 100 == 0:
                    print(f"Checked: {total_checked}/729, Valid found: {len(valid_gates)}")

        valid_gates_list = [{'and_gate': dict(and_pairs), 'or_gate': dict(or_pairs)}
                            for and_pairs, or_pairs in valid_gates]

        print(f"\nTotal checked: {total_checked}")
        print(f"Valid unique combinations found: {len(valid_gates_list)}")
        return valid_gates_list


    def save_gates(self, valid_gates, filename="ternary_gates.json"):
        serializable_gates = [
            {
                'and_gate': {f"{k[0]},{k[1]}": v for k, v in gate_pair['and_gate'].items()},
                'or_gate': {f"{k[0]},{k[1]}": v for k, v in gate_pair['or_gate'].items()}
            }
            for gate_pair in valid_gates
        ]
        with open(filename, 'w') as f:
            json.dump(serializable_gates, f, indent=2)
        print(f"Saved {len(valid_gates)} valid gate combinations to {filename}")

    def display_sample_gates(self, valid_gates, num_samples=5):
        print("\n=== SAMPLE VALID GATES ===")
        for i, gate_pair in enumerate(valid_gates[:num_samples]):
            print(f"\n--- Gate Pair {i + 1} ---")
            print("AND Gate:")
            for a in self.values:
                for b in self.values:
                    result = gate_pair['and_gate'][(a, b)]
                    print(f"  {a:2} AND {b:2} = {result:2}")
            print("OR Gate:")
            for a in self.values:
                for b in self.values:
                    result = gate_pair['or_gate'][(a, b)]
                    print(f"  {a:2} OR  {b:2} = {result:2}")

    # def analyze_unknown_behaviors(self, valid_gates):
    #     """Analyze different behaviors for unknown value combinations"""
    #     if not valid_gates:
    #         return
    #
    #     print("\n=== ANALYSIS OF UNKNOWN VALUE BEHAVIORS ===")
    #
    #     behaviors = {}
    #     for gate_pair in valid_gates:
    #         and_gate = gate_pair['and_gate']
    #         or_gate = gate_pair['or_gate']
    #
    #         # Focus on unknown combinations
    #         behavior = []
    #         for combo in [(0, 0), (0, 1), (0, -1)]:
    #             and_result = and_gate[combo]
    #             or_result = or_gate[combo]
    #             behavior.append((and_result, or_result))
    #
    #         behavior_key = tuple(behavior)
    #         if behavior_key not in behaviors:
    #             behaviors[behavior_key] = []
    #         behaviors[behavior_key].append(gate_pair)
    #
    #     print(f"Found {len(behaviors)} distinct behavioral patterns:")
    #     for i, (behavior, gates) in enumerate(behaviors.items()):
    #         print(f"\nPattern {i + 1} ({len(gates)} gates):")
    #         print("  (0,0)  (0,1)  (0,-1)")
    #         print(f"AND: {behavior[0][0]:2}     {behavior[1][0]:2}     {behavior[2][0]:2}")
    #         print(f"OR:  {behavior[0][1]:2}     {behavior[1][1]:2}     {behavior[2][1]:2}")


if __name__ == "__main__":
    generator = TernaryLogicGenerator()
    print("=== PRACTICAL TERNARY LOGIC GATE GENERATOR ===")
    print("Values: -1 (False), 0 (Unknown), 1 (True)")

    # Then try with monotonicity
    print("\n" + "=" * 50)
    valid_gates = generator.generate_valid_gates()

    if valid_gates:
        generator.save_gates(valid_gates, "ternary_gates_constrained.json")
        generator.display_sample_gates(valid_gates)
        generator.analyze_unknown_behaviors(valid_gates)
        print(f"\n=== SUMMARY ===")
        print(f"With monotonicity: {len(valid_gates)} combinations")
