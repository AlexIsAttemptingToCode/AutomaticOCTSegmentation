"""evaluation meterics for boundary segmentation """
import numpy as np


def dice_score(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    
    """Dice similarity coefficient between two binary masks

        Dice = 2 |A ∩ B| / (|A| + |B|)

    Range [0, 1]; with 1 = perfect agreement

    """
    a = (mask_pred > 0)
    b = (mask_gt > 0)
    s = a.sum() + b.sum()

    if s == 0:
        return 1.0
    
    return float(2.0 * np.logical_and(a, b).sum() / s)


def mean_absolute_surface_distance(boundary_pred: np.ndarray,
                                   boundary_gt: np.ndarray) -> float:
    
    """Percolumn mean |pred - gt| row distance

    ignore any nan cols

    """

    diff = np.abs(boundary_pred.astype(float) - boundary_gt.astype(float))
    valid = ~np.isnan(diff)

    if valid.sum() == 0:
        return float("nan")
    
    return float(diff[valid].mean())


def max_surface_distance(boundary_pred: np.ndarray,
                         boundary_gt: np.ndarray) -> float:
    
    """max per col |pred - gt|, for boundaries"""
    
    diff = np.abs(boundary_pred.astype(float) - boundary_gt.astype(float))
    valid = ~np.isnan(diff)

    if valid.sum() == 0:
        return float("nan")
    
    return float(diff[valid].max())


def evaluate(boundary_pred: np.ndarray,
             boundary_gt: np.ndarray,
             mask_pred: np.ndarray | None = None,
             mask_gt: np.ndarray | None = None) -> dict:
    
    """compute metrics for slice"""

    out = {
        "MASD_px": mean_absolute_surface_distance(boundary_pred, boundary_gt),
        "MaxSD_px": max_surface_distance(boundary_pred, boundary_gt),
    }

    if mask_pred is not None and mask_gt is not None:
        out["Dice"] = dice_score(mask_pred, mask_gt)

    return out
