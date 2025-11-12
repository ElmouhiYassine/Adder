import matplotlib.pyplot as plt
import json

# === your adder imports go here ===

# Load JSON data
with open('uncertainty_results_vectors_10_trits_sampled.json', 'r') as file:
    data = json.load(file)

# Collect avg_max values for all adder types across uncertainty levels
adder_types = [
    "Strong-Kleene", "Pessimistic-Optimistic", "Sobocinski",
    "Bochvar external", "Sette", "Lukasiewicz", "Gaines-Rescher"
]
variances = {adder: [] for adder in adder_types}

for i in range(11):
    for adder in adder_types:
        # Accessing the nested structure for each adder
        adder_data = data[f'{i}'][adder]
        # Extracting the 'avg' value from max
        avg_max = adder_data["max"]["avg"]
        avg_min = adder_data["min"]["avg"]
        avg_mid = adder_data["mid"]["avg"]
        avg_var_max = adder_data["min"]["var"]
        variances[adder].append(round(avg_var_max, 2))

def plot_variances(variances):
    n = 11
    levels = range(n)
    plt.figure(figsize=(11, n))
    for name, vals in variances.items():
        plt.plot(levels, vals, marker="o", label=name)

    plt.xlabel("Uncertainty Level")
    plt.ylabel("Variance")
    plt.title("Variance vs. Uncertainty Level (Min Deviation), 10-trits inputs, 200 000 samples")
    plt.legend()
    plt.grid(True)

    # Force x-axis ticks to be integers only
    plt.xticks(levels)

    plt.show()

# Call the plotting function
plot_variances(variances)