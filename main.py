import numpy as np
import matplotlib.pyplot as plt

from HTLConvolution.dataset_loader import load_xmnist_first_N
from HTLConvolution.helpers import (
    encode_uint8_to_UV,
    decode_Y_numeric_center,
    conv2d_valid_int,
)
from HTLConvolution.htl_convolution import convolution

from Adders.sobocinski_adder import sobocinski_ripple   # default adder
from Adders.balanced_ternary_adder import balanced_ternary_add
from Adders.lukasiewicz_adder import luka_ripple_add
from Adders.bochvar_external_adder import bochvar_ripple_add
from Adders.sette_adder import sette_ripple_add
from Adders.gaines_rescher_adder import gaines_ripple_add
# ============================================================
#        HTL CONVOLUTION BENCHMARK
# ============================================================

def run_htl_convolution_benchmark(
    num_images=100,
    num_kernels=10,
    K1=5,
    K2=10,
    kernel_size=3,
    noise_levels=None,
    dataset="mnist",
    seed=1234,

    # pass any uncertain binary adder here
    uncertain_adder=sobocinski_ripple,
):

    if noise_levels is None:
        noise_levels = [0, 5, 10, 15, 20, 25, 30, 31]

    rng = np.random.default_rng(seed)

    # Load a single global set of images
    images = load_xmnist_first_N(dataset, n_samples=num_images).astype(int)

    # ---- storage for final MAE curves ----
    mean_naive_mae = {σ: [] for σ in noise_levels}
    mean_htl_mae = {σ: [] for σ in noise_levels}

    # ---- global statistics ----
    total_kernel_wins = 0
    total_kernel_losses = 0
    kernel_loss_cases = []

    best_win = {"gain": -np.inf}
    smallest_win = {"gain": np.inf}

    # ============================================================
    # Loop over kernels
    # ============================================================
    for kernel_index in range(num_kernels):

        # Random kernel in {-1, 0, +1}
        kernel = rng.integers(-1, 2, size=(kernel_size, kernel_size))
        print("Kernel:")
        print(kernel)

        # MAE over ALL IMAGES for this kernel
        kernel_naive_mae = []
        kernel_htl_mae = []

        for img_index, img in enumerate(images):

            for σ in noise_levels:

                # Noise
                noise = rng.normal(0, σ, size=img.shape).round().astype(int)
                noise_abs = np.abs(noise)
                img_noisy = np.clip(img + noise, 0, 255).astype(int)

                # Clean reference conv
                clean_conv = conv2d_valid_int(img, kernel)

                # Naive conv
                naive_conv = conv2d_valid_int(img_noisy, kernel)
                naive_mae = np.mean(np.abs(naive_conv - clean_conv))

                # HTL encoding + convolution
                X, _, _ = encode_uint8_to_UV(img_noisy, noise_abs, K1, K2)

                Y = convolution(
                    X,
                    kernel,
                    uncertain_adder,            # <<--- NEW ADER HERE
                    balanced_ternary_add,       # V always uses BT adder
                    K1=K1,
                    K2=K2,
                )

                htl_map = decode_Y_numeric_center(Y, K1, K2)
                htl_mae = np.mean(np.abs(htl_map - clean_conv))

                # Per-kernel tracking
                kernel_naive_mae.append(naive_mae)
                kernel_htl_mae.append(htl_mae)

                # Global per-noise tracking
                mean_naive_mae[σ].append(naive_mae)
                mean_htl_mae[σ].append(htl_mae)

                # Track best win / smallest win (per-image)
                gain = naive_mae - htl_mae

                if gain > best_win["gain"]:
                    best_win = {
                        "gain": gain,
                        "kernel": kernel.tolist(),
                        "sigma": σ,
                        "image_index": img_index,
                        "naive_mae": naive_mae,
                        "htl_mae": htl_mae,
                    }

                if gain > 0 and gain < smallest_win["gain"]:
                    smallest_win = {
                        "gain": gain,
                        "kernel": kernel.tolist(),
                        "sigma": σ,
                        "image_index": img_index,
                        "naive_mae": naive_mae,
                        "htl_mae": htl_mae,
                    }

        # ============================================================
        #           END OF KERNEL — Evaluate Mean Win/Loss
        # ============================================================
        mean_naive_k = np.mean(kernel_naive_mae)
        mean_htl_k = np.mean(kernel_htl_mae)

        if mean_htl_k <= mean_naive_k:
            total_kernel_wins += 1
        else:
            total_kernel_losses += 1

            print("\n KERNEL LOSS DETECTED — HTL WORSE ON AVERAGE")
            print("Kernel:")
            print(kernel)
            print(f"Mean Naive MAE: {mean_naive_k:.6f}")
            print(f"Mean HTL MAE:   {mean_htl_k:.6f}")
            print(f"Difference:      {mean_htl_k - mean_naive_k:.6f}")
            print("-" * 60)

            kernel_loss_cases.append({
                "kernel": kernel.tolist(),
                "mean_naive_mae": mean_naive_k,
                "mean_htl_mae": mean_htl_k,
                "difference": mean_htl_k - mean_naive_k,
            })

    # ============================================================
    # Aggregate final MAE curves per noise level
    # ============================================================
    naive_curve = [np.mean(mean_naive_mae[σ]) for σ in noise_levels]
    htl_curve = [np.mean(mean_htl_mae[σ]) for σ in noise_levels]

    # ============================================================
    # Plot
    # ============================================================
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, naive_curve, marker="o", label="Naive")
    plt.plot(noise_levels, htl_curve, marker="s", label="HTL Framework")
    plt.xlabel("Noise σ")
    plt.ylabel("Mean MAE across all kernels & images")
    plt.title("HTL Convolution Benchmark (Kernel-Level Evaluation)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "mean_naive_curve": naive_curve,
        "mean_htl_curve": htl_curve,
        "total_kernel_wins": total_kernel_wins,
        "total_kernel_losses": total_kernel_losses,
        "kernel_loss_cases": kernel_loss_cases,
        "best_win": best_win,
        "smallest_win": smallest_win,
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    results = run_htl_convolution_benchmark(uncertain_adder=sobocinski_ripple)
    print("\n===== HTL Benchmark Results =====")
    print("Kernel Wins:", results["total_kernel_wins"])
    print("Kernel Losses:", results["total_kernel_losses"])
    print("Best Win Case:", results["best_win"])
    print("Smallest Win Case:", results["smallest_win"])


""" How to use different uncertain adders
you can do:

from Adders.the_adder import the_adder
run_htl_convolution_benchmark(uncertain_adder=the_adder)
"""