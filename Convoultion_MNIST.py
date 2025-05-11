import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import fetch_openml
from cyclical_adder import cyclical_adder
from strong_kleene import strong_kleene_full_adder
import matplotlib.colors as mcolors

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
    return images[indices[0]]
def binarize_image(image, threshold=128):
    return (image > threshold).astype(int)

# Function to ternarize image ( 1 --> 1, 0 --> -1)
def ternarize(image):
    return np.where(image == 1, 1, -1)


def add_value_to_sum(sum_trits, v):
    carry = -1  # Initial carry
    for i in range(4):  # Loop over the 4 trits of the sum
        a = sum_trits[i]
        b = v if i == 0 else -1  # Add v only to the least significant trit
        cin = carry
        sum_trits[i], carry = strong_kleene_full_adder(a, b, cin)
    return sum_trits

def ternary_sum(products):
    sum_trits = [-1, -1, -1, -1]  # Start with 0
    for v in products:  # Loop over all 9 products
        sum_trits = add_value_to_sum(sum_trits.copy(), v)
    return sum_trits


# Function to apply convolution
def convolve_ternary(image, kernel):
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
            sum_vec = ternary_sum(flat_products.copy())
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
def visualize_results(ter_original, noisy_example, kernel, ternary_cmap, norm):
    plt.figure(figsize=(12, 8))

    # Row 1: Original and Noisy Images
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(ter_original, cmap='hot')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Noisy Image (Example)")
    plt.imshow(noisy_example, cmap='hot')
    plt.axis('off')

    # Row 2: Convolution of Original and Noisy Images
    plt.subplot(2, 2, 3)
    plt.title("Convolution of Original Image")
    plt.imshow(convolve_ternary(ter_original, kernel), cmap='hot')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Convolution of Noisy Image")
    plt.imshow(convolve_ternary(noisy_example, kernel), cmap=ternary_cmap, norm=norm)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('convolution_results.png')
# Main code
if __name__ == "__main__":
    kernels = [
        np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]]),
        np.ones((3, 3), dtype=int),
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    ]
    digit = 2
    image = select_image(digit)
    binarized_image = binarize_image(image)
    tern = ternarize(binarized_image)
    noisy_example = add_noise(tern, p=0.1)
    out  = convolve_ternary(noisy_example, kernels[0])
    has_unknown = np.any(out == 0, axis=-1)
    print(out)
    # count how many output pixels have at least one unknown
    n_unknown = np.count_nonzero(has_unknown)
    total = has_unknown.size

    print(f"{n_unknown} out of {total} output vectors contain at least one 0 (unknown).")