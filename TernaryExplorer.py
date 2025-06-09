import itertools
import json
from typing import Dict, Tuple, List


class TernaryLogicGenerator:
    def __init__(self):
        self.values = [-1, 0, 1]

    def neg(self, x):
        return -x if x != 0 else 0

    def is_valid_gates(self, and_gate, or_gate):
        for a in self.values:
            for b in self.values:
                if self.neg(or_gate[(a, b)]) != and_gate[(self.neg(a), self.neg(b))]:
                    return False
        return True

    def generate_valid_gates(self):
        and_known = {(1, 1): 1, (-1, -1): -1, (1, -1): -1, (-1, 1): -1}
        or_known = {(1, 1): 1, (-1, -1): -1, (1, -1): 1, (-1, 1): 1}

        unique_pairs = [(0, 0), (0, 1), (0, -1)]
        #Exploring 3^3 = 27 possibilities per gate, total 729 combinations

        valid_gates = set()
        for and_vals in itertools.product(self.values, repeat=3):
            for or_vals in itertools.product(self.values, repeat=3):
                and_gate = and_known.copy()
                or_gate = or_known.copy()

                for i, (a, b) in enumerate(unique_pairs):
                    and_gate[(a, b)] = and_vals[i]
                    and_gate[(b, a)] = and_vals[i]
                    or_gate[(a, b)] = or_vals[i]
                    or_gate[(b, a)] = or_vals[i]

                if self.is_valid_gates(and_gate, or_gate):
                    gate_tuple = (frozenset(and_gate.items()), frozenset(or_gate.items()))
                    valid_gates.add(gate_tuple)

        valid_gates_list = [{'and_gate': dict(and_pairs), 'or_gate': dict(or_pairs)}
                            for and_pairs, or_pairs in valid_gates]
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

    def display_sample_gates(self, valid_gates, num_samples=3):
        print("\n=== SAMPLE VALID GATES ===")
        for i, gate_pair in enumerate(valid_gates[:num_samples]):
            print(f"\n--- Gate Pair {i + 1} ---")
            print("AND Gate:")
            for a in self.values:
                for b in self.values:
                    result = gate_pair['and_gate'][(a, b)]
                    print(f"  {a} AND {b} = {result}")
            print("OR Gate:")
            for a in self.values:
                for b in self.values:
                    result = gate_pair['or_gate'][(a, b)]
                    print(f"  {a} OR {b} = {result}")


if __name__ == "__main__":
    generator = TernaryLogicGenerator()
    print("=== TERNARY LOGIC GATE GENERATOR ===")
    print("Values: -1 (False), 0 (Unknown), 1 (True)")
    valid_gates = generator.generate_valid_gates()
    if valid_gates:
        generator.save_gates(valid_gates)
        generator.display_sample_gates(valid_gates)
        print(f"\n=== SUMMARY ===")
        print(f"Found {len(valid_gates)} unique valid AND-OR gate combinations")
    else:
        print("No valid gate combinations found!")