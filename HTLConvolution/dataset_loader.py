import numpy as np

import numpy as np


def load_xmnist_first_N(dataset: str = "mnist", n_samples: int = 64) -> np.ndarray:
    """
    Load first N images from MNIST, Fashion-MNIST, or KMNIST.
    Returns uint8 images (N, H, W).
    """
    try:
        from torchvision import transforms

        # Select the dataset class based on the string
        if dataset.lower() == "mnist":
            from torchvision.datasets import MNIST as DS
        elif dataset.lower() == "fashion":
            from torchvision.datasets import FashionMNIST as DS
        elif dataset.lower() == "kmnist":
            from torchvision.datasets import KMNIST as DS
        else:
            raise ValueError("dataset must be 'mnist', 'fashion', or 'kmnist'")

        # Load the dataset
        ds = DS(root="./data", train=True, download=True,
                transform=transforms.ToTensor())

        imgs = []
        for k in range(min(n_samples, len(ds))):
            x, _ = ds[k]
            # Convert (1, H, W) float tensor to (H, W) uint8 numpy array
            arr = (x.numpy()[0] * 255.0).round().astype(np.uint8)
            imgs.append(arr)

        return np.stack(imgs, axis=0)

    except Exception as e:
        print(f"Could not load dataset '{dataset}': {e}. Returning random noise.")
        rng = np.random.default_rng(0)
        return rng.integers(0, 256, size=(n_samples, 28, 28), dtype=np.uint8)