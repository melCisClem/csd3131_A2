"""Step (a): energy map.

energy[i, j] = |dI/dx| + |dI/dy| using the 3x3 Sobel filter on the grayscale
image. Low energy = the pixel looks like its neighbours = cheap to remove.
"""
import cv2
import numpy as np


def compute_energy(img: np.ndarray) -> np.ndarray:
    # Return a (height, width) float64 energy map for a grayscale or colour image.
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]              # BGRA/RGBA: drop the alpha (transparency) channel
    grayscale = img.astype(np.float64)
    if grayscale.ndim == 3:
        grayscale = grayscale.mean(axis=2)  # average the colour channels; their order (BGR/RGB) doesn't matter
    gradient_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    gradient_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    return np.abs(gradient_x) + np.abs(gradient_y)
