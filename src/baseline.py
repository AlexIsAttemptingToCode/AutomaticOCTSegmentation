""" naive baselines for ILM segmentation, for contrasting against main dijkstra """

import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter1d


def baseline_otsu(img: np.ndarray) -> np.ndarray:

    """per col -> first foreground pixel from top after Otsu thresholding.

    simplest approach: threshold the image into background / retina, then in each column take the topmost foreground row as the ILM.
    no constraints
    """

    H, W = img.shape
    _, binar = cv.threshold(img, 0, 255,
                            cv.THRESH_BINARY + cv.THRESH_OTSU)
    boundary = np.full(W, H - 1, dtype=np.int32)

    for c in range(W):
        idx = np.where(binar[:, c] > 0)[0]

        if idx.size:
            boundary[c] = idx[0]
    return boundary


def baseline_gradient_argmax(img: np.ndarray,
                             smooth_col: int = 21,
                             smooth_row: int = 0) -> np.ndarray:
    
    """per col argmax of the positive vertical Sobel gradient.

    
    same cost as graph method but with no shortest pathj, each column moves as it pleases to find optimum
   
    """

    H, W = img.shape

    blurred = cv.GaussianBlur(img, (1, 2 * smooth_col + 1), 0).astype(float)

    grad = np.diff(blurred, axis=0)

    grad[grad < 0] = 0

    boundary = np.argmax(grad, axis=0)

    if smooth_row > 0:
        boundary = gaussian_filter1d(boundary.astype(float),
                                     sigma=smooth_row).round().astype(np.int32)
    return boundary.astype(np.int32)
