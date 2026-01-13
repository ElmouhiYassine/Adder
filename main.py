import numpy as np
import matplotlib.pyplot as plt
import json

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
    K1=5,
    K2=10,
    noise_levels=None,
    dataset="mnist",
    seed=1234,
    uncertain_adder=sobocinski_ripple,
):

    if noise_levels is None:
        noise_levels = [0, 5, 10, 15, 20, 25, 30, 31]

    rng = np.random.default_rng(seed)

    images = load_xmnist_first_N(dataset, n_samples=num_images).astype(int)

    mean_naive_mae = {σ: [] for σ in noise_levels}
    mean_htl_mae = {σ: [] for σ in noise_levels}

    total_kernel_wins = 0
    total_kernel_losses = 0
    kernel_loss_cases = []

    best_win = {"gain": -np.inf}
    smallest_win = {"gain": np.inf}

    # ============================================================
    #   LOOP OVER ALL REAL TERNARY KERNELS
    # ============================================================
    for kernel_name, kernel in TERNARY_KERNELS:

        print(f"\n--- Testing kernel: {kernel_name} ---")

        kernel_np = np.array(kernel, dtype=int)  # <--- FIX HERE

        kernel_naive_mae = []
        kernel_htl_mae = []

        for img_index, img in enumerate(images):

            for σ in noise_levels:

                noise = rng.normal(0, σ, size=img.shape).round().astype(int)
                noise_abs = np.abs(noise)
                img_noisy = np.clip(img + noise, 0, 255).astype(int)

                clean_conv = conv2d_valid_int(img, kernel_np)

                naive_conv = conv2d_valid_int(img_noisy, kernel_np)
                naive_mae = np.mean(np.abs(naive_conv - clean_conv))

                X, _, _ = encode_uint8_to_UV(img_noisy, noise_abs, K1, K2)

                Y = convolution(
                    X,
                    kernel_np,
                    uncertain_adder,
                    balanced_ternary_add,
                    K1=K1,
                    K2=K2,
                )

                htl_map = decode_Y_numeric_center(Y, K1, K2)
                htl_mae = np.mean(np.abs(htl_map - clean_conv))

                kernel_naive_mae.append(naive_mae)
                kernel_htl_mae.append(htl_mae)

                mean_naive_mae[σ].append(naive_mae)
                mean_htl_mae[σ].append(htl_mae)

                gain = naive_mae - htl_mae

                if gain > best_win["gain"]:
                    best_win = {
                        "gain": gain,
                        "kernel": kernel_name,
                        "sigma": σ,
                        "image_index": img_index,
                        "naive_mae": naive_mae,
                        "htl_mae": htl_mae,
                    }

                if 0 < gain < smallest_win["gain"]:
                    smallest_win = {
                        "gain": gain,
                        "kernel": kernel_name,
                        "sigma": σ,
                        "image_index": img_index,
                        "naive_mae": naive_mae,
                        "htl_mae": htl_mae,
                    }

        # mean performance for this kernel
        # mean performance for this kernel
        mean_naive_k = np.mean(kernel_naive_mae)
        mean_htl_k = np.mean(kernel_htl_mae)

        # ================================================
        #   PLOT FOR THIS KERNEL (NEW!)
        # ================================================
        plt.figure(figsize=(9, 5))
        plt.plot(noise_levels, [np.mean(kernel_naive_mae[i::len(noise_levels)])
                                for i in range(len(noise_levels))],
                 marker="o", label="Naive")

        plt.plot(noise_levels, [np.mean(kernel_htl_mae[i::len(noise_levels)])
                                for i in range(len(noise_levels))],
                 marker="s", label=f"HTL ({adder_name})")

        plt.xlabel("Noise σ")
        plt.ylabel("MAE for this kernel")
        plt.title(f"Kernel: {kernel_name} – Adder: {adder_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        fname = f"kernel_plot_{adder_name}_{kernel_name}.png"
        plt.savefig(fname, dpi=300)
        plt.show()
        plt.close()

        print(f"Saved per-kernel plot: {fname}")

        if mean_htl_k <= mean_naive_k:
            total_kernel_wins += 1
        else:
            total_kernel_losses += 1
            kernel_loss_cases.append({
                "kernel": kernel_name,
                "mean_naive_mae": mean_naive_k,
                "mean_htl_mae": mean_htl_k,
                "difference": mean_htl_k - mean_naive_k,
            })

    # ============================================================
    # Aggregate final MAE curves per noise level
    # ============================================================
    naive_curve = [np.mean(mean_naive_mae[σ]) for σ in noise_levels]
    htl_curve = [np.mean(mean_htl_mae[σ]) for σ in noise_levels]

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

    uncertain_adders = {
        "sobocinski": sobocinski_ripple,
        #"lukasiewicz": luka_ripple_add,
        # "bochvar": bochvar_ripple_add,
        # "sette": sette_ripple_add,
        # "gaines_rescher": gaines_ripple_add,
    }


    TERNARY_KERNELS = [
        # --- 1. Directional Gradients (Edge Detection) ---
        ("gradient_x", [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        ("gradient_y", [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),

        # --- 2. Laplacian (Feature/Point Detection) ---
        # 4-connectivity (Cross pattern)
        ("laplacian_4", [[0, -1, 0], [-1, 1, -1], [0, -1, 0]]),
        # 8-connectivity (Square pattern - identical to High Pass)
        ("laplacian_8", [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]]),

        # --- 3. Diagonal / Anisotropic ---
        # Detects diagonal edges (Emboss effect)
        ("emboss_diag", [[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),

        # --- 4. Smoothing / Integration ---
        # (Previously Box/Gaussian - identical in {-1,0,1})
        ("box_blur", [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),

        # --- 5. Control ---
        # Identity kernel to test signal preservation
        ("identity", [[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    ]
    noise_levels = [0, 5, 10, 15, 20, 25, 30, 31]

    for adder_name, adder_func in uncertain_adders.items():
        print(f"\n========== Testing with adder: {adder_name} ==========\n")

        results = run_htl_convolution_benchmark(
            uncertain_adder=adder_func,
            num_images=100,
            dataset="mnist"
        )

        # Save results
        with open(f"htl_results_{adder_name}.json", "w") as f:
            json.dump(results, f, indent=2)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(noise_levels, results["mean_naive_curve"], marker="o", label="Naive")
        plt.plot(noise_levels, results["mean_htl_curve"], marker="s",
                 label=f"HTL ({adder_name})")

        plt.xlabel("Noise σ")
        plt.ylabel("Mean MAE across all kernels & images")
        plt.title(f"HTL Convolution Benchmark – {adder_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plot_{adder_name}.png", dpi=300)
        plt.show()
