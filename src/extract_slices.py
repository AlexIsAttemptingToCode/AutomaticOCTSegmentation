"""extract annotated B-scans and ground-truth ILM boundaries from one Duke SD-OCT subject (.mat) and save as PNGs and .npz"""


from pathlib import Path
import numpy as np
import scipy.io
import cv2 as cv

ROOT = Path(__file__).resolve().parent.parent
MAT_PATH = ROOT / "data" / "raw" / "2015_BOE_Chiu" / "Subject_02.mat"
SLICE_DIR = ROOT / "data" / "slices"
GT_DIR = ROOT / "data" / "ground_truth"
SLICE_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    m = scipy.io.loadmat(MAT_PATH)
    volume = m["images"]            # (H, W, N) uint8
    layers1 = m["manualLayers1"]    # (8, W, N) float
    layers2 = m["manualLayers2"]    # (8, W, N) float

    # pick slices that have ILM (layer 0) 2 manual annotations by experts

    ilm1 = layers1[0]   # (W, N)
    ilm2 = layers2[0]

    N = ilm1.shape[1]

    annotated = [
        i for i in range(N)
        if np.any(~np.isnan(ilm1[:, i])) and np.any(~np.isnan(ilm2[:, i]))
    ]

    print(f"annotated slices: {annotated} (n={len(annotated)})")

    meta = []

    for k, idx in enumerate(annotated):
        img = volume[:, :, idx]
        out_png = SLICE_DIR / f"slice_{k:02d}.png"
        cv.imwrite(str(out_png), img)
        meta.append({"i": k, "subject_slice_index": int(idx),
                     "shape": img.shape})

    # save the ILM ground truth (per-column row indices, nan where missing)
    ilm_gt1 = np.stack([ilm1[:, i] for i in annotated], axis=0)   # (n_slices, W)
    ilm_gt2 = np.stack([ilm2[:, i] for i in annotated], axis=0)

    np.savez(GT_DIR / "ilm_boundaries.npz",
             expert1=ilm_gt1, expert2=ilm_gt2,
             slice_indices=np.array(annotated))
    
    print("saved", GT_DIR / "ilm_boundaries.npz",
          "shapes:", ilm_gt1.shape, ilm_gt2.shape)

    # save full slice indices and meta 
    np.savez(GT_DIR / "meta.npz",
             slice_indices=np.array(annotated),
             total_slices=N,
             image_shape=np.array(volume.shape[:2]))


if __name__ == "__main__":
    main()
