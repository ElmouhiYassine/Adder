import numpy as np
import import_ipynb
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import fetch_openml
from cyclical_adder import cyclical_full_adder
from strong_kleene import strong_kleene_full_adder
import matplotlib.colors as mcolors
from preprocessing_majority import majority_impute, add_block_noise
from preprocessing_unanimous import unanimous_impute
from SK_Quasi_adder import map_quasi_adder
from post_processing import interval_range_classification, min_assumption, max_assumption
from pessimistic_adder import pessimistic_full_adder
from NormalAdder import Normal_adder

ternary_cmap = mcolors.ListedColormap(['black', (1.0, 0.4, 0.0), 'white'])
bounds = [-1.5, -0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, ternary_cmap.N)



# Function to select an MNIST image of a specific digit
def select_image(digit):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    images = mnist.data.reshape(-1, 28, 28)  # Reshape to 28x28
    labels = mnist.target.astype(int)  # Convert labels to integers

    # Select the first image of the specified digit
    indices = np.where(labels == digit)[0]
    n = len(indices)
    var = np.random.randint(0, n)
    return images[indices[var]]
def binarize_image(image, threshold=128):
    return (image > threshold).astype(int)

# Function to ternarize image ( 1 --> 1, 0 --> -1)
def ternarize(image):
    return np.where(image == 1, 1, -1)


def add_value_to_sum(sum_trits, v, adder):
    carry = -1  # Initial carry
    for i in range(4):  # Loop over the 4 trits of the sum
        a = sum_trits[i]
        b = v if i == 0 else -1  # Add v only to the least significant trit
        cin = carry
        # here the adder @@@@@2
        sum_trits[i], carry = adder(a, b, cin)
    return sum_trits

def ternary_sum(products, adder):
    sum_trits = [-1, -1, -1, -1]  # Start with 0
    for v in products:  # Loop over all 9 products
        sum_trits = add_value_to_sum(sum_trits.copy(), v, adder)
    return sum_trits


# Function to apply convolution
def convolve_ternary(image, kernel, adder):
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    output = np.zeros((out_h, out_w, 4), dtype=int)


    for i in range(out_h):
        for j in range(out_w):
            window = image[i:i + kh, j:j + kw]
            mask = (kernel != 0)
            products = window[mask] * kernel[mask]
            flat_products = products.flatten()
            sum_vec = ternary_sum(flat_products.copy(), adder)
            output[i, j, :] = sum_vec.copy()

    return output



def add_noise(ter, p=0.1):
    noisy = ter.copy()
    h, w = noisy.shape
    num_flips = int(p * h * w)
    indices = np.random.choice(h * w, num_flips, replace=False)
    for idx in indices:
        i = idx // w
        j = idx % w
        noisy[i, j] = 0
    return noisy

# Function to visualize results (2 rows only)
def visualize_results(ter_original, noisy_example, imputed, kernel, ternary_cmap, norm, adder):
    """
    Display:
      Row 1: original, noisy, imputed
      Row 2: convolution of original, noisy, imputed
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Row 1
    axes[0, 0].imshow(ter_original, cmap='hot', vmin=-1, vmax=1)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_example, cmap='hot', vmin=-1, vmax=1)
    axes[0, 1].set_title("Noisy Image")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(imputed, cmap='hot', vmin=-1, vmax=1)
    axes[0, 2].set_title("Imputed Image")
    axes[0, 2].axis('off')

    # Row 2
    conv_orig = convolve_ternary(ter_original, kernel, adder)
    mapped_out = interval_range_classification(conv_orig)
    axes[1, 0].imshow(mapped_out, cmap='hot', vmin=-1, vmax=1)
    axes[1, 0].set_title("Conv. Original")
    axes[1, 0].axis('off')

    conv_noisy = convolve_ternary(noisy_example, kernel, adder)
    mapped_out2 = interval_range_classification(conv_noisy)
    axes[1, 1].imshow(mapped_out2, cmap=ternary_cmap, norm=norm)
    axes[1, 1].set_title("Conv. Noisy")
    axes[1, 1].axis('off')

    conv_imputed = convolve_ternary(imputed, kernel, adder)
    mapped_out3 = interval_range_classification(conv_imputed)
    axes[1, 2].imshow(mapped_out3, cmap=ternary_cmap, norm=norm)
    axes[1, 2].set_title("Conv. Imputed")
    axes[1, 2].axis('off')


    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.3, wspace=0.2)
    plt.show()

# Main code
if __name__ == "__main__":
    kernels = [
        np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]]),
        np.ones((3, 3), dtype=int),
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    ]
    digit = 3
    image = select_image(digit)
    binarized_image = binarize_image(image)
    tern = ternarize(binarized_image)
    noisy_example = add_noise(tern, p=0.3)
    imputed_image = majority_impute(noisy_example)
    out  = convolve_ternary(noisy_example, kernels[0], cyclical_full_adder)

    visualize_results(tern,noisy_example,imputed_image,kernels[0],ternary_cmap,norm , cyclical_full_adder)

    has_unknown = np.any(out == 0, axis=-1)
    # count how many output pixels have at least one unknown
    n_unknown = np.count_nonzero(has_unknown)
    total = has_unknown.size

    print(f"{n_unknown} out of {total} output vectors contain at least one 0 (unknown).")
