"""Graph-theoretic ILM segmentation for scans.

ILM = topmost retinal boundary with sharp dark-to-bright transition

model the image as a directed graph:
    nodes  = pixels (r, c)
    edges  = (r, c) -> (r', c+1) for r' in {r-1, r, r+1}
    weight = pixel cost at (r', c+1)

per pixel cost rewards strong dark-above / bright-below transitions
(the ILM signature) by inverting the vertical Sobel gradient.

the minimum cost path from any pixel in column 0 to any pixel in the last
column traces the ILM

"""
from __future__ import annotations

import heapq
import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter1d


def compute_cost_image(img: np.ndarray) -> np.ndarray:

    """
    per pixel traversal cost in [0, 1], low cost = strong ILM-like edge
    ILM -> pixel above is dark, pixel below is bright 
    (retina) -> vertical gradient dI/dr is large and pos, reward by inverting norm gradient

    """
    img_f = img.astype(np.float64)
    sobel_y = cv.Sobel(img_f, cv.CV_64F, 0, 1, ksize=5)

    grad = sobel_y.copy()
    grad[grad < 0] = 0
    g_min, g_max = grad.min(), grad.max()
    grad = (grad - g_min) / (g_max - g_min + 1e-9)

    cost = 1.0 - grad
    return cost


def dijkstra_layer(cost: np.ndarray,
                   row_band: tuple[int, int] | None = None,
                   per_column_band: np.ndarray | None = None) -> np.ndarray:
    
    """
    shortest left -> right path through `cost` , returns row indices (W,)

    `row_band`: optional global (r_min, r_max) to restrict search
    `per_column_band`: optional (W, 2) array giving (r_min, r_max) per col
        override `row_band` when provided -> pixels outside the column band excluded from the search

    """

    H, W = cost.shape
    if per_column_band is not None:
        assert per_column_band.shape == (W, 2)
        col_lo = np.clip(per_column_band[:, 0], 0, H).astype(int)
        col_hi = np.clip(per_column_band[:, 1], 0, H).astype(int)
    else:
        if row_band is None:
            r_lo, r_hi = 0, H
        else:
            r_lo, r_hi = row_band
            r_lo = max(0, r_lo)
            r_hi = min(H, r_hi)
        col_lo = np.full(W, r_lo, dtype=int)
        col_hi = np.full(W, r_hi, dtype=int)

    INF = np.inf
    dist = np.full((H, W), INF)
    prev = np.full((H, W), -1, dtype=np.int32)

    heap: list[tuple[float, int, int]] = []
    for r in range(col_lo[0], col_hi[0]):
        dist[r, 0] = cost[r, 0]
        heapq.heappush(heap, (cost[r, 0], r, 0))

    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        if c == W - 1:
            continue
        nc = c + 1
        for dr in (-1, 0, 1):
            nr = r + dr
            if not (col_lo[nc] <= nr < col_hi[nc]):
                continue
            nd = d + cost[nr, nc]
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                prev[nr, nc] = r
                heapq.heappush(heap, (nd, nr, nc))

    # end at lowest cost reached pixel in last col
    last = dist[:, W - 1].copy()
    if np.all(np.isinf(last)):
        return np.zeros(W, dtype=np.int32)
    end_r = int(np.argmin(last))
    boundary = np.zeros(W, dtype=np.int32)
    r = end_r
    for c in range(W - 1, -1, -1):
        boundary[c] = r
        if c > 0:
            r = int(prev[r, c])
    return boundary


def estimate_row_band(img: np.ndarray,
                      half_height: int = 80) -> tuple[int, int]:
    
    """

    return one global horizontal band (compress rows)
    
    """
    H, _ = img.shape

    prior = column_ilm_prior(img)
    center = int(np.median(prior))
    return max(0, center - half_height), min(H, center + half_height)


def _detect_top_artifact(img: np.ndarray) -> int:

    """
    
    return first row that contains real image 

    exclude any bad cropping (white artifact of some sort top left in slices)

    """
    row_var = img.std(axis=1)

    # first row with variation
    above = row_var > 5
    if not above.any():
        return 0
    return int(np.argmax(above))


def column_ilm_prior(img: np.ndarray,
                     smooth_col: int = 21,
                     smooth_row: int = 41,
                     skip_buffer: int = 5) -> np.ndarray:
    
    """
    per colILM row estimate

    ILM first dark to bright, smooth each col and take first pos gradient peak/col (remove outliers) and interpolate 

    """
    H, W = img.shape
    first_real = _detect_top_artifact(img)
    start = first_real + skip_buffer

    blurred = cv.GaussianBlur(img, (1, 2 * smooth_col + 1), 0).astype(float)
    grad_y = np.diff(blurred, axis=0)
    grad_y[grad_y < 0] = 0
    if start > 0:
        grad_y[:start] = 0   # ignore any artifact gradients above this row

    col_max = grad_y.max(axis=0)
    cand = np.full(W, np.nan)
    for c in range(W):
        if col_max[c] < 1.0:
            continue
        thr = 0.5 * col_max[c]
        idx = np.where(grad_y[:, c] >= thr)[0]
        if idx.size:
            cand[c] = idx[0]

    valid = ~np.isnan(cand)
    if valid.sum() < W * 0.1:
        return np.full(W, H // 4, dtype=float)

    # reject outliers vs global median
    med = np.nanmedian(cand)
    mad = np.nanmedian(np.abs(cand - med)) + 1.0
    cand[np.abs(cand - med) > 6 * mad] = np.nan

    xs = np.arange(W)
    valid = ~np.isnan(cand)
    if valid.sum() < 2:
        return np.full(W, med, dtype=float)
    interp = np.interp(xs, xs[valid], cand[valid])
    interp = gaussian_filter1d(interp, sigma=smooth_row / 3.0)
    return interp


def segment_ilm(img_preprocessed: np.ndarray,
                smooth_sigma: float = 3.0,
                use_prior: bool = True,
                prior_halfband: int = 25
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    
    ILM segmentation

    strat:
        compute the perpixel cost image

        est. a per col ILM prior 

        run Dijkstra restricted to ± `prior_halfband` pixels around the prior in each column (anchor to region)

        gaussian smooth resulting boundary

    return

        boundary (W,) int row indices, smoothed
        cost     (H, W) cost image used by Dijkstra

    """
    H, W = img_preprocessed.shape
    cost = compute_cost_image(img_preprocessed)

    if use_prior:
        prior = column_ilm_prior(img_preprocessed)
        col_band = np.stack([prior - prior_halfband,
                             prior + prior_halfband + 1], axis=1)
        boundary = dijkstra_layer(cost, per_column_band=col_band)
    else:
        boundary = dijkstra_layer(cost)

    boundary_smooth = gaussian_filter1d(boundary.astype(float),
                                        sigma=smooth_sigma)
    boundary_smooth = np.clip(np.rint(boundary_smooth), 0,
                              H - 1).astype(np.int32)
    return boundary_smooth, cost


def compute_cost_image_bright_to_dark(img: np.ndarray) -> np.ndarray:
    """
    
    cost image reward bright to dark vertical transitions

    mirror `compute_cost_image` for inverse boundaries (RPE)

    """
    img_f = img.astype(np.float64)
    sobel_y = cv.Sobel(img_f, cv.CV_64F, 0, 1, ksize=5)
    grad = -sobel_y               # negative gradient = bright above, dark below
    grad[grad < 0] = 0
    g_min, g_max = grad.min(), grad.max()
    grad = (grad - g_min) / (g_max - g_min + 1e-9)
    return 1.0 - grad


def segment_layer_below(img: np.ndarray,
                        upper_boundary: np.ndarray,
                        min_gap: int = 10,
                        max_gap: int = 200,
                        polarity: str = "dark_to_bright",
                        smooth_sigma: float = 3.0) -> np.ndarray:
    

    """
    
    segment one retinal layer lies below upper boundary

    args
        img:             preprocessed scan
        upper_boundary:  row indices (W,) of boundary above target
        min_gap:         min vert dist below upper boundary
        max_gap:         max vert dist below upper boundary
        polarity:        "dark_to_bright" or "bright_to_dark"
        smooth_sigma:    final Gaussian smoothing of the boundary

    search range per column is [upper(c)+min_gap, upper(c)+max_gap]

    """
    H, W = img.shape

    if polarity == "dark_to_bright":
        cost = compute_cost_image(img)
    elif polarity == "bright_to_dark":
        cost = compute_cost_image_bright_to_dark(img)
    else:
        raise ValueError(f"unknown polarity: {polarity}")

    col_band = np.zeros((W, 2), dtype=int)
    col_band[:, 0] = np.clip(upper_boundary + min_gap, 0, H - 1)
    col_band[:, 1] = np.clip(upper_boundary + max_gap, 1, H)

    # guarantee at least one valid row/col
    col_band[:, 1] = np.maximum(col_band[:, 1], col_band[:, 0] + 1)

    boundary = dijkstra_layer(cost, per_column_band=col_band)
    boundary = gaussian_filter1d(boundary.astype(float), sigma=smooth_sigma)
    boundary = np.clip(np.rint(boundary), 0, H - 1).astype(np.int32)
    return boundary


def segment_three_layers(img: np.ndarray) -> dict:

    """
    
    segment ILM, IS/OS and RPE/Choroid boundary

    retur dict {"ILM": ..., "IS_OS": ..., "RPE": ...}, each a (W,) row array -> use sequential Dijkstra with exclusion bands so that later boundaries cannot cross earlier ones

    """
    ilm, _ = segment_ilm(img)
    is_os = segment_layer_below(img, ilm,
                                min_gap=40, max_gap=130,
                                polarity="dark_to_bright")
    rpe = segment_layer_below(img, is_os,
                              min_gap=4, max_gap=40,
                              polarity="bright_to_dark")
    return {"ILM": ilm, "IS_OS": is_os, "RPE": rpe}


def boundary_to_mask(boundary: np.ndarray, image_shape: tuple[int, int],
                     band: int = 2) -> np.ndarray:
    """
    
    convert per col row array into binary band mask

    """

    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for c in range(W):
        r = int(boundary[c])
        r0 = max(0, r - band)
        r1 = min(H, r + band + 1)
        mask[r0:r1, c] = 255
    return mask


def nan_boundary_to_mask(boundary_nan: np.ndarray,
                         image_shape: tuple[int, int],
                         band: int = 2) -> np.ndarray:
    
    """

     `boundary_to_mask` but skips cols where boundary is NaN
    
    """

    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for c in range(W):
        v = boundary_nan[c]
        if np.isnan(v):
            continue
        r = int(round(v))
        r0 = max(0, r - band)
        r1 = min(H, r + band + 1)
        mask[r0:r1, c] = 255
    return mask
