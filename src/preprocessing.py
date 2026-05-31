""" preprocessing for SD-OCT B-scans: speckle denoising + contrast boost"""
import numpy as np
import cv2 as cv


def preprocess_oct(img: np.ndarray) -> np.ndarray:
    """denoise and enhance B-scan

    Pipeline:
        1. Median filter (3x3): suppresses solo speckle pixels
        2. Bilateral filter: edge-preserving smoothing 
        3. CLAHE: boosts contrast to sharpen transitions
    """

    if img.dtype != np.uint8:
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    img = cv.medianBlur(img, 3)
    img = cv.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return img


def preprocess_steps(img: np.ndarray) -> dict:

    """return each stage of `preprocess_oct`"""

    if img.dtype != np.uint8:
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    out = {"raw": img.copy()}

    out["median"] = cv.medianBlur(img, 3)

    out["bilateral"] = cv.bilateralFilter(out["median"],
                                          d=7, sigmaColor=50, sigmaSpace=50)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    out["clahe"] = clahe.apply(out["bilateral"])
    
    return out
