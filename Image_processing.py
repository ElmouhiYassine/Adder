import numpy as np
import import_ipynb
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import fetch_openml
from cyclical_adder import cyclical_full_adder
from pesi_op_adder import full_op_adder
from pesi_op_adder import full_op_adder
from strong_kleene import strong_kleene_full_adder
import matplotlib.colors as mcolors
from preprocessing_majority import majority_impute, add_block_noise
from preprocessing_unanimous import unanimous_impute
from SK_Quasi_adder import map_quasi_adder
from post_processing import interval_range_classification, min_assumption, max_assumption
from pessimistic_adder import pessimistic_full_adder
from NormalAdder import Normal_adder
from Convoultion_MNIST import ternarize, select_image, binarize_image, add_noise

import numpy as np
import matplotlib.pyplot as plt


def image_add_subtract(image, adder_func):
    h, w = image.shape
    result = np.zeros((h, w), dtype=int)

    for i in range(h):
        for j in range(w):
            a = int(image[i, j])
            # First A + A
            s1, _ = adder_func(a, a, -1)
            # Then s1 - A  (i.e. s1 + (-a))
            s2, _ = adder_func(s1, -a, 1)
            result[i, j] = s2

    return result


def plot_results(original, result, title):

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original, cmap='hot', vmin=-1, vmax=1)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(result, cmap='hot', vmin=-1, vmax=1)
    axs[1].set_title(f"Result: {title}")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


def process_all_adders(image):

    results = [
        ("Strong Kleene", image_add_subtract(image, strong_kleene_full_adder)),
        ("Quasi", image_add_subtract(image, map_quasi_adder)),
        ("Pessi-Opti", image_add_subtract(image, full_op_adder))
    ]

    for title, result in results:
        plot_results(image, result, title)


image = select_image(2)
bin_img = binarize_image(image)
ternarize_img = ternarize(bin_img)
noise = add_noise(ternarize_img)
process_all_adders(noise)
