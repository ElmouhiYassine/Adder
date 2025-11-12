import numpy as np
import matplotlib.pyplot as plt
from Adders.pesi_op_adder import full_op_adder
from Adders.strong_kleene import strong_kleene_full_adder
import matplotlib.colors as mcolors
from Adders.SK_Quasi_adder import map_quasi_adder
from Convoultion_MNIST import ternarize, select_image, binarize_image, add_noise

def image_add_subtract(image, adder_func, flag_value=0):
    h, w = image.shape
    result = np.zeros((h, w), dtype=float)  # Use float to support flag_value=2

    for i in range(h):
        for j in range(w):
            a = int(image[i, j])
            # Step 1: A + A, keep the carry
            s1, carry1 = adder_func(a, a, -1)  # Start with carry-in 0
            # Step 2: s1 - A (i.e., s1 + (-a)), keep the final carry
            s2, carry2 = adder_func(s1, -a, carry1)  # Propagate carry
            # Check if result is correct (s2 == A and carry2 == 0)
            if carry2 == -1:
                result[i, j] = s2
            else:
                result[i, j] = flag_value  # Flag incorrect result

    return result


def plot_results(original, result, title):
    # Define custom colormap: -1 (black), 0 (gray), 1 (white), 2 (red)
    cmap = mcolors.ListedColormap(['black', 'orange', 'white', 'red'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]  # Boundaries for -1, 0, 1, 2
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot original image without vmin/vmax
    axs[0].imshow(original, cmap=cmap, norm=norm)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # Plot result image without vmin/vmax
    axs[1].imshow(result, cmap=cmap, norm=norm)
    axs[1].set_title(f"Result: {title}")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

def process_all_adders(image, flag_value=0):
    results = [
        ("Strong Kleene", image_add_subtract(image, strong_kleene_full_adder, flag_value)),
        ("Quasi", image_add_subtract(image, map_quasi_adder, flag_value)),
        ("Pessi-Opti", image_add_subtract(image, full_op_adder, flag_value))
    ]

    for title, result in results:
        plot_results(image, result, title)

# Test the code
image = select_image(2)
bin_img = binarize_image(image)
ternarize_img = ternarize(bin_img)
noise = add_noise(ternarize_img)
process_all_adders(noise, flag_value=0)  # Flag errors as 0 (gray)
# Optional: process_all_adders(noise, flag_value=2)  # Flag errors as 2 (red)