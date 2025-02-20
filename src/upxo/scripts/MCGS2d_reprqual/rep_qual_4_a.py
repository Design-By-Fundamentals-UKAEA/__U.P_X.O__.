# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:03:57 2024

@author: rg5749
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

def compute_psd(image):
    # Compute the 2D Fourier transform of the image
    f_transform = fft2(image)
    # Shift the zero frequency component to the center
    f_transform_shifted = fftshift(f_transform)
    # Compute the power spectrum
    power_spectrum = np.abs(f_transform_shifted)**2
    # Normalize the power spectrum
    psd = power_spectrum / np.sum(power_spectrum)
    return psd

# Load your images (ensure they are grayscale)
image1 = plt.imread(r'C:\Development\M2MatMod\upxo_packaged\upxo_private\src\upxo\scripts\MCGS2d_reprqual\Picture1.png')
image2 = plt.imread(r'C:\Development\M2MatMod\upxo_packaged\upxo_private\src\upxo\scripts\MCGS2d_reprqual\Picture2.png')

# Compute the PSD for each image
psd1 = compute_psd(image1)
psd2 = compute_psd(image2)

# Plot the PSDs for visual comparison
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(np.log(psd1 + 1e-8)[0], cmap='gray')  # Use log scale for better visualization
plt.title('PSD of Grain Structure 1')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(np.log(psd2 + 1e-8), cmap='gray')  # Use log scale for better visualization
plt.title('PSD of Grain Structure 2')
plt.colorbar()

plt.show()

# Quantitative comparison (e.g., correlation)
similarity = np.corrcoef(psd1.flatten(), psd2.flatten())[0, 1]
print(f'Similarity (correlation coefficient): {similarity}')
