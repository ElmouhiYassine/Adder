import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

from HTLConvolution.dataset_loader import load_xmnist_first_N
from HTLConvolution.helpers import (
    encode_uint8_to_UV,
    decode_Y_numeric_center,
    conv2d_valid_int,
)
from HTLConvolution.htl_convolution import convolution

from Adders.sobocinski_adder import sobocinski_ripple,map_quasi_adder   # default adder
from Adders.balanced_ternary_adder import balanced_ternary_add
from Adders.lukasiewicz_adder import luka_ripple_add
from Adders.bochvar_external_adder import bochvar_ripple_add
from Adders.sette_adder import sette_ripple_add
from Adders.gaines_rescher_adder import gaines_ripple_add
from Adders.strong_kleene_adder import SK_ripple_add
# ============================================================
#        HTL CONVOLUTION BENCHMARK
# ============================================================
uncertain_adders = {
        "SK": SK_ripple_add,
        "Sobo": sobocinski_ripple,
        "Luka": luka_ripple_add,
        "Bochvar": bochvar_ripple_add,
        "Sette": sette_ripple_add,
        "Gaines": gaines_ripple_add,
    }
def run_htl_convolution_benchmark_table(
    num_images=100,
    K1=5,
    K2=10,
    noise_levels=None,
    dataset="mnist",
    base_seed=1234,
    n_noise_realizations=5,
    adders=uncertain_adders,
):

    if noise_levels is None:
        noise_levels = [0, 5, 10, 15, 20, 25, 30, 31]

    if adders is None:
        adders = {
            "SK": SK_ripple_add,
            "Sobo": sobocinski_ripple,
            "Luka": luka_ripple_add,
        }

    images = load_xmnist_first_N(dataset, n_samples=num_images).astype(int)

    naive_samples = {σ: [] for σ in noise_levels}
    htl_samples = {name: {σ: [] for σ in noise_levels} for name in adders}


    for kernel_name, kernel in TERNARY_KERNELS:
        print(kernel_name)
        kernel_np = np.array(kernel, dtype=int)

        for img_idx, img in enumerate(images):

            for noise_id in range(n_noise_realizations):


                seed = (
                    base_seed
                    + 10_000 * img_idx
                    + 1_000 * noise_id
                )
                rng = np.random.default_rng(seed)

                for σ in noise_levels:

                    noise = rng.normal(0, σ, size=img.shape).astype(int)
                    noise_abs = np.abs(noise)
                    img_noisy = np.clip(img + noise, 0, 255).astype(int)

                    clean_conv = conv2d_valid_int(img, kernel_np)

                    # -------- Naive --------
                    naive_conv = conv2d_valid_int(img_noisy, kernel_np)
                    naive_mae = float(np.mean(np.abs(naive_conv - clean_conv)))
                    naive_samples[σ].append(naive_mae)

                    # -------- HTL (per adder) --------
                    X, _, _ = encode_uint8_to_UV(img_noisy, noise_abs, K1, K2)

                    for name, uncertain_adder in adders.items():
                        Y = convolution(
                            X,
                            kernel_np,
                            uncertain_adder,
                            balanced_ternary_add,
                            K1=K1,
                            K2=K2,
                        )
                        htl_map = decode_Y_numeric_center(Y, K1, K2)
                        htl_mae = float(np.mean(np.abs(htl_map - clean_conv)))
                        htl_samples[name][σ].append(htl_mae)

    rows = []
    for σ in noise_levels:
        row = {
            "Uncertainty Level": σ,
            "Avg Error Naive": np.mean(naive_samples[σ]),
            "SD Error Naive":  np.std(naive_samples[σ], ddof=0),
        }

        for name in uncertain_adders.keys():
            vals = htl_samples[name][σ]
            row[f"Avg Error {name}"] = np.mean(vals)
            row[f"SD Error {name}"]  = np.std(vals, ddof=0)

        rows.append(row)

    return pd.DataFrame(rows)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    uncertain_adders = {
        "SK": SK_ripple_add,
        "Sobo": sobocinski_ripple,
        "Luka": luka_ripple_add,
        "Bochvar": bochvar_ripple_add,
        "Sette": sette_ripple_add,
        "Gaines": gaines_ripple_add,
    }

    # TERNARY_KERNELS = [
    #     ("all_ones", [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    #     ("cross", [[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    #     ("diamond", [[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
    #     ("x_pattern", [[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
    #     ("corners", [[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
    #     ("horizontal", [[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
    #     ("vertical", [[0, 1, 0], [0, 1, 0], [0, 1, 0]]),
    #     ("diagonal_main", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #     ("diagonal_anti", [[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    #     ("plus_sign", [[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    #     ("top_half", [[1, 1, 1], [1, 0, 1], [0, 0, 0]]),
    #     ("bottom_half", [[0, 0, 0], [1, 0, 1], [1, 1, 1]]),
    #     ("neighbors_8", [[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
    #     ("edge_frame", [[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
    # ]
    TERNARY_KERNELS = [
        ("gradient_x", [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        ("gradient_y", [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        ("laplacian_4", [[0, -1, 0], [-1, 1, -1], [0, -1, 0]]),
        ("laplacian_8", [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]]),
        ("roberts_cross", [[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
        ("robinson_ne", [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]),
        ("robinson_nw", [[1, 1, 0], [1, 0, -1], [0, -1, -1]]),
    ]


    df = run_htl_convolution_benchmark_table(
        num_images=100,
        dataset="mnist",
        noise_levels=[0, 5, 10, 15, 20, 25, 30, 31],
        n_noise_realizations=5,
        base_seed=1234,
        adders=uncertain_adders,
    )


    pd.set_option("display.float_format", lambda x: f"{x:8.4f}")
    print("\n===== HTL Convolution Benchmark Results =====\n")
    print(df.to_string(index=False))


    df.to_csv("htl_negative_benchmark_results.csv", index=False)
    print("\nSaved results to: htl_benchmark_results.csv")

    # noise_levels = [0, 5, 10, 15, 20, 25, 30, 31]
    #
    # for adder_name, adder_func in uncertain_adders.items():
    #     print(f"\n========== Testing with adder: {adder_name} ==========\n")
    #
    #     results = run_htl_convolution_benchmark(
    #         uncertain_adder=adder_func,
    #         num_images=100,
    #         dataset="mnist"
    #     )
    #
    #     # Save results
    #     with open(f"htl_results_{adder_name}.json", "w") as f:
    #         json.dump(results, f, indent=2)
    #
    #     # Plot
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(noise_levels, results["mean_naive_curve"], marker="o", label="Naive")
    #     plt.plot(noise_levels, results["mean_htl_curve"], marker="s",
    #              label=f"HTL ({adder_name})")
    #
    #     plt.xlabel("Noise σ")
    #     plt.ylabel("Mean MAE across all kernels & images")
    #     plt.title(f"HTL Convolution Benchmark – {adder_name}")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(f"plot_{adder_name}.png", dpi=300)
    #     plt.show()
