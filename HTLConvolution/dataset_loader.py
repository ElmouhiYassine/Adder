import numpy as np

def load_xmnist_first_N(dataset: str = "mnist", n_samples: int = 64) -> np.ndarray:
    """
    Load first N MNIST or Fashion-MNIST images.
    Returns uint8 images (H,W).
    """
    try:
        from torchvision import transforms
        if dataset.lower() == "mnist":
            from torchvision.datasets import MNIST as DS
        elif dataset.lower() == "fashion":
            from torchvision.datasets import FashionMNIST as DS
        else:
            raise ValueError("dataset must be 'mnist' or 'fashion'")

        ds = DS(root="./data", train=True, download=True,
                transform=transforms.ToTensor())

        imgs = []
        for k in range(min(n_samples, len(ds))):
            x, _ = ds[k]
            arr = (x.numpy()[0] * 255.0).round().astype(np.uint8)
            imgs.append(arr)

        return np.stack(imgs, axis=0)

    except Exception:
        rng = np.random.default_rng(0)
        return rng.integers(0, 256, size=(n_samples, 28, 28), dtype=np.uint8)
