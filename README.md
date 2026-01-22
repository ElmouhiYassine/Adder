# Hybrid Ternary Logic for Uncertainty-Aware Arithmetic

This repository accompanies the paper **“Hybrid Ternary Logic for Uncertainty-Aware Arithmetic”** and provides the full experimental framework used to study uncertainty-aware arithmetic units and their application to convolution layers in convolutional neural networks.

The code explores **Hybrid Ternary Logic (HyTL)**, which combines explicit uncertainty modeling with the arithmetic advantages of balanced ternary representations, and evaluates several ternary adders under controlled uncertainty.

This repository is intended to support the experimental results of the paper.

---

## Repository Structure

### [`Adders/`](Adders/)

This package contains implementations of all ternary and uncertainty-aware adders used in the paper, including:

- Strong Kleene  
- Łukasiewicz  
- Sobociński  
- Bochvar external  
- Sette  
- Gaines–Rescher  
- Balanced ternary adder  

These adders are used both in standalone arithmetic evaluations and as building blocks for convolution operations.

---

### [`ternary_arithmetic/`](ternary_arithmetic/)
This package contains the core arithmetic study tools. It includes:
 
- Arithmetic operations (addition, negation, bounds computation)  
- Evaluation utilities for minimum, maximum, and midpoint approximation under uncertainty  

This module supports the arithmetic analysis presented in the paper.

---

### [`HyTLConvolution/`](HyTLConvolution/)
This package implements the **HyTL convolution framework**, applying hybrid ternary logic (HyTL) to the convolution layer.

It includes:
- A HyTL-based convolution operator  
- Dataset loaders (MNIST, Fashion-MNIST)  
- Helper functions  

The convolution layer uses uncertainty-aware adders in the lower (uncertain) part and balanced ternary arithmetic in the upper (noise-free) part, as described in the paper.

---

### `main.py`
This file includes the script that runs **HyTL benchmarking experiments** on different datasets using different kernels.

- Compares Naive convolution with HyTL convolution  
- Evaluates different uncertainty-aware adders  
- Runs experiments across multiple noise levels, kernels, and random seeds  
- Exports results as CSV files for analysis and visualization  

---
