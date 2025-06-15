
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import f1_score
import json

# Import adders
from pesi_op_adder import full_op_adder
from strong_kleene import strong_kleene_full_adder
from SK_Quasi_adder import map_quasi_adder
from cyclical_adder import cyclical_full_adder
from NormalAdder import Normal_adder

# Import image processing functions
from Convoultion_MNIST import (
    select_image, binarize_image, ternarize,
    convolve_ternary, add_noise
)
from Ternary_New_Adder import (
get_Adder
)
from post_processing import (
min_assumption, max_assumption, interval_range_classification
)
from preprocessing_min_max_avg import (
Max_values,
Min_values,
)

# all adders to evaluate
ADDER_FUNCTIONS = {
    "pessimistic-opti": full_op_adder,
    "kleene_strong": strong_kleene_full_adder,
    "quasi": map_quasi_adder
}
for i in range(6):
    ADDER_FUNCTIONS[f'adder_{i}'] = get_Adder(i)



def compute_ground_truth(image, kernel, adder):
    binarized = binarize_image(image)
    ternarized = ternarize(binarized)
    return convolve_ternary(ternarized, kernel, adder)


def evaluate_adder(adder_func, kernel, noise_level, trials=10):
    metrics_accum = {
        "vector_accuracy": 0,
        "trit_accuracy": 0,
        "sign_accuracy": 0,
        "f1_score": 0,
        "uncertainty_rate": 0,
        "mse": 0,
        "mae": 0
    }

    for _ in range(trials):
        # Select a random digit
        digit = np.random.randint(0, 10)
        image = select_image(digit)

        # Binarize and ternarize
        binarized = binarize_image(image)
        ternarized = ternarize(binarized)
        noisy = add_noise(ternarized, p=noise_level)

        # Max assumption
        # max_noisy = np.where(noisy == 0, 1, noisy)

        # Convolutions
        conv_original = compute_ground_truth(image, kernel, Normal_adder)
        conv_noisy = convolve_ternary(noisy, kernel, adder_func)

        # Flatten
        flat_noisy = conv_noisy.reshape(-1)
        flat_original = conv_original.reshape(-1)

        # Metrics
        vector_match = np.all(conv_noisy == conv_original, axis=-1)
        vector_accuracy = np.mean(vector_match)

        trit_accuracy = np.mean(flat_noisy == flat_original)
        sign_accuracy = np.mean(np.sign(flat_noisy) == np.sign(flat_original))
        uncertainty_rate = np.mean(flat_noisy == 0)

        try:
            f1 = f1_score(flat_original, flat_noisy, average='weighted', zero_division=0)
        except:
            f1 = 0

        mse = np.mean((flat_noisy - flat_original) ** 2)
        mae = np.mean(np.abs(flat_noisy - flat_original))

        # Accumulate
        metrics_accum["vector_accuracy"] += vector_accuracy
        metrics_accum["trit_accuracy"] += trit_accuracy
        metrics_accum["sign_accuracy"] += sign_accuracy
        metrics_accum["f1_score"] += f1
        metrics_accum["uncertainty_rate"] += uncertainty_rate
        metrics_accum["mse"] += mse
        metrics_accum["mae"] += mae

    # Average over trials
    averaged_metrics = {key: val / trials for key, val in metrics_accum.items()}
    averaged_metrics["adder_name"] = adder_func.__name__

    return averaged_metrics



def compute_ternary_metrics(prediction, ground_truth):
    """Compute metrics for ternary output arrays (H x W)"""
    # Flatten arrays
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()

    # Basic metrics
    accuracy = float(np.mean(pred_flat == gt_flat))  # Convert to float

    # F1 score (weighted average for ternary classes)
    try:
        f1 = float(f1_score(gt_flat, pred_flat, average='weighted', zero_division=0))
    except:
        f1 = 0.0

    # Uncertainty rate
    uncertainty_rate = float(np.mean(pred_flat == 0))

    # Confusion matrix components - convert to native Python int
    tp = int(np.sum((gt_flat == 1) & (pred_flat == 1)))
    tn = int(np.sum((gt_flat == -1) & (pred_flat == -1)))
    fp = int(np.sum((gt_flat == -1) & (pred_flat == 1)))
    fn = int(np.sum((gt_flat == 1) & (pred_flat == -1)))
    uc = int(np.sum(pred_flat == 0))  # Uncertain predictions

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "uncertainty_rate": uncertainty_rate,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "uncertain": uc
    }


def run_comparison(kernel, noise_levels=[0.1, 0.3, 0.5], output_file="results.json"):

    results = defaultdict(dict)

    for noise in noise_levels:
        print(f"\n=== Evaluating at noise level: {noise * 100}% ===")
        for adder_name, adder_func in ADDER_FUNCTIONS.items():
            print(f"Testing {adder_name}...")
            metrics = evaluate_adder(adder_func, kernel, noise)
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
            # Extract metric values across noise levels
            values = [results[noise][adder][metric]
                      for noise in sorted(results.keys())]
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
    """Print formatted comparison table"""
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
    DIGIT = 3
    KERNEL = np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]])
    NOISE_LEVELS = [0.1, 0.3, 0.5]

    image = select_image(DIGIT)

    results = run_comparison( KERNEL, NOISE_LEVELS)

    visualize_results(results)

    # Print summary table
    print_summary_table(results)

    # run_comparison(image, KERNEL, NOISE_LEVELS)