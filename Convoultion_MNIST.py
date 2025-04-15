import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import fetch_openml
from cyclical_adder import cyclical_adder


# Function to select an MNIST image of a specific digit
def select_image(digit):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    images = mnist.data.reshape(-1, 28, 28)  # Reshape to 28x28
    labels = mnist.target.astype(int)  # Convert labels to integers

    # Select the first image of the specified digit
    indices = np.where(labels == digit)[0]
    return images[indices[0]]  # 28x28 grayscale image


# Function to binarize image
def binarize_image(image, threshold=128):
    return (image > threshold).astype(int)


# Function to add noise by flipping bits
def add_noise(binary_image, p=0.1):
    noisy = binary_image.copy()
    h, w = binary_image.shape
    num_flips = int(p * h * w)
    indices = np.random.choice(h * w, num_flips, replace=False)
    for idx in indices:
        i = idx // w
        j = idx % w
        noisy[i, j] = 1 - noisy[i, j]
    return noisy


# Function to ternarize image ( 1 --> 1, 0 --> -1)
def ternarize(image):
    return np.where(image == 1, 1, -1)


# Function to sum ternary values
def ternary_sum(values):
    total = [0]
    carry = 0
    for v in values:
        temp_sum, temp_carry = cyclical_adder(total, [v])
        total = temp_sum

        carry_sum, carry_carry = cyclical_adder([carry], [temp_carry])
        carry = carry_sum[0]
        carry += carry_carry
    while carry != 0:
        temp_sum, temp_carry = cyclical_adder(total, [carry])
        total = temp_sum
        carry = temp_carry
    return total[0]


# Function to apply convolution
def convolve_ternary(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    output = np.zeros((out_h, out_w), dtype=int)

    for i in range(out_h):
        for j in range(out_w):
            window = image[i:i + kh, j:j + kw]
            products = window * kernel
            flat_products = products.flatten()
            sum_value = ternary_sum(flat_products)
            output[i, j] = 1 if sum_value > 0 else -1 if sum_value < 0 else 0

    return output


# Function to analyze uncertainty
def analyze_uncertainty(outputs):
    h, w = outputs[0].shape
    variance_map = np.zeros((h, w))
    frequency_map = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            values = [output[i, j] for output in outputs]
            mean = np.mean(values)
            variance = np.mean([(v - mean) ** 2 for v in values])
            variance_map[i, j] = variance
            counts = Counter(values)
            mode_count = max(counts.values())
            frequency_map[i, j] = mode_count / len(outputs)

    return variance_map, frequency_map


# Function to run experiments
def run_experiments(image, kernel, num_runs=1000, p_noise=0.1):
    outputs = []
    for _ in range(num_runs):
        binary_noisy = add_noise(image, p_noise)
        ternary_noisy = ternarize(binary_noisy)
        output = convolve_ternary(ternary_noisy, kernel)
        outputs.append(output)
    return outputs


# Function to visualize results
def visualize_results(original, noisy_example, variance_map, frequency_map):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap='binary')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Noisy Image (Example)")
    plt.imshow(noisy_example, cmap='binary')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Variance Map")
    plt.imshow(variance_map, cmap='hot')
    plt.colorbar(label='Variance')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Mode Frequency Map")
    plt.imshow(frequency_map, cmap='viridis')
    plt.colorbar(label='Frequency')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('convolution_results.png')


# Main code
if __name__ == "__main__":
    digit = 2
    image = select_image(digit)
    binary_image = binarize_image(image)

    kernel = np.array([
        [0, 1, 0],
        [1, -1, 1],
        [0, 1, 0]
    ])

    outputs = run_experiments(binary_image, kernel)
    variance_map, frequency_map = analyze_uncertainty(outputs)
    noisy_example = add_noise(binary_image)

    visualize_results(binary_image, noisy_example, variance_map, frequency_map)