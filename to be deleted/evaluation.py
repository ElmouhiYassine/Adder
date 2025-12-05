import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import f1_score
import json

# Import adders
from Adders.pesi_op_adder import full_op_adder
from Adders.strong_kleene_adder import strong_kleene_full_adder
from Adders.SK_Quasi_adder import map_quasi_adder
from Adders.normal_adder import Normal_adder

# Import image processing functions
from Convoultion_MNIST import (
    select_image, binarize_image, ternarize,
    convolve_ternary, add_noise
)
from Adders.Ternary_New_Adder import (
    get_Adder
)

# All adders to evaluate
ADDER_FUNCTIONS = {"pessimistic-opti": full_op_adder, "kleene_strong": strong_kleene_full_adder,
                   "quasi": map_quasi_adder, f'Super_Pessimistic_Optimistic': get_Adder(0),
                   f'Collapsible': get_Adder(2), "Triangular" : get_Adder(4),"Bi-Triangular":get_Adder(5) }


def compute_ground_truth(image, kernel, adder):
    binarized = binarize_image(image)
    ternarized = ternarize(binarized)
    return convolve_ternary(ternarized, kernel, adder)


def evaluate_adder(adder_func, kernel, trial_data, ground_truths):
    metrics_accum = {
        "vector_accuracy": 0,
        "trit_accuracy": 0,
        "sign_accuracy": 0,
        "f1_score": 0,
        "uncertainty_rate": 0,
        "mse": 0,
        "mae": 0
    }
    num_trials = len(trial_data)
    for (_, noisy), gt in zip(trial_data, ground_truths):
        # Convolve with noisy input
        conv_noisy = convolve_ternary(noisy, kernel, adder_func)
        flat_noisy = conv_noisy.reshape(-1)
        flat_gt = gt.reshape(-1)

        # Compute metrics
        vector_match = np.all(conv_noisy == gt, axis=-1)
        vector_accuracy = np.mean(vector_match)
        trit_accuracy = np.mean(flat_noisy == flat_gt)
        sign_accuracy = np.mean(np.sign(flat_noisy) == np.sign(flat_gt))
        uncertainty_rate = np.mean(flat_noisy == 0)
        try:
            f1 = f1_score(flat_gt, flat_noisy, average='weighted', zero_division=0)
        except:
            f1 = 0
        mse = np.mean((flat_noisy - flat_gt) ** 2)
        mae = np.mean(np.abs(flat_noisy - flat_gt))

        # Accumulate
        metrics_accum["vector_accuracy"] += vector_accuracy
        metrics_accum["trit_accuracy"] += trit_accuracy
        metrics_accum["sign_accuracy"] += sign_accuracy
        metrics_accum["f1_score"] += f1
        metrics_accum["uncertainty_rate"] += uncertainty_rate
        metrics_accum["mse"] += mse
        metrics_accum["mae"] += mae

    # Average over trials
    averaged_metrics = {key: val / num_trials for key, val in metrics_accum.items()}
    averaged_metrics["adder_name"] = adder_func.__name__
    return averaged_metrics


def run_comparison(kernel, noise_levels=[0.1, 0.3, 0.5], trials=10, output_file="results.json"):
    results = defaultdict(dict)

    # Generate a fixed set of base trials (digits and images)
    base_trials = []
    for _ in range(trials):
        digit = np.random.randint(0, 10)
        image = select_image(digit)
        binarized = binarize_image(image)
        ternarized = ternarize(binarized)
        base_trials.append((image, ternarized))

    # Compute ground truth once using clean images
    ground_truths = [compute_ground_truth(image, kernel, Normal_adder) for image, _ in base_trials]

    # Evaluate for each noise level
    for noise in noise_levels:
        print(f"\n=== Evaluating at noise level: {noise * 100}% ===")
        # Generate noisy versions for this noise level
        noisy_trials = [(image, add_noise(ternarized, p=noise)) for image, ternarized in base_trials]

        for adder_name, adder_func in ADDER_FUNCTIONS.items():
            print(f"Testing {adder_name}...")
            metrics = evaluate_adder(adder_func, kernel, noisy_trials, ground_truths)
            results[noise][adder_name] = metrics

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    return results


def visualize_results(results):
    metrics = ["vector_accuracy", "f1_score", "mse", "uncertainty_rate"]
    titles = ["Vector Accuracy", "F1 Score", "Mean Squared Error", "Uncertainty Rate"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    for i, metric in enumerate(metrics):
        ax = axs[i]
        for adder in ADDER_FUNCTIONS.keys():
            values = [results[noise][adder][metric] for noise in sorted(results.keys())]
            ax.plot(sorted(results.keys()), values, 'o-', label=adder)
        ax.set_title(titles[i])
        ax.set_xlabel("Noise Level")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.savefig("adder_comparison.png")
    plt.show()


def print_summary_table(results):
    print("\n" + "=" * 60)
    print("Adder Performance Summary")
    print("=" * 60)
    print(f"{'Adder':<15} {'Noise':<6} {'Vec Acc':<8} {'F1':<8} {'MSE':<10} {'Uncertainty':<12}")
    print("-" * 60)
    for noise in sorted(results.keys()):
        for adder in ADDER_FUNCTIONS.keys():
            m = results[noise][adder]
            print(f"{adder:<15} {noise:<6.2f} {m['vector_accuracy']:<8.4f} "
                  f"{m['f1_score']:<8.4f} {m['mse']:<10.4f} {m['uncertainty_rate']:<12.4f}")
        print("-" * 60)


if __name__ == "__main__":
    # Configuration
    KERNEL = np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]])
    ker2 =    np.ones((3, 3), dtype=int)
    NOISE_LEVELS = [0.1, 0.3, 0.5]
    TRIALS = 100
    kernels = [
        np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]]),
        np.ones((3, 3), dtype=int),
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    ]

    results = run_comparison(kernels[2], NOISE_LEVELS, TRIALS)
    visualize_results(results)
    print_summary_table(results)