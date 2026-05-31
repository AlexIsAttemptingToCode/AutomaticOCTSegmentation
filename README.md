# Retinal Layer Segmentation in SD-OCT with Graph Theory

Graph-theoretic segmentation of retinal layer boundaries in Spectral-Domain
Optical Coherence Tomography (SD-OCT) B-scans. Each B-scan is modelled as a
directed graph and layer boundaries are recovered via Dijkstra's shortest-path
algorithm. The approach is extended from a single layer (ILM) to three layers
(ILM, IS/OS junction, RPE/Bruch's membrane) via sequential constrained
Dijkstra.

Course project: **MSLS / CO4**, FHNW, SS 2026.

---

## Results

On the 11 expert-annotated slices of `Subject_02`:

| method            | MASD mean [px] | MASD [µm] |
|-------------------|---------------:|----------:|
| Graph (Dijkstra)  | **1.47**       | **5.7**   |
| Otsu (top pixel)  | 62.34          | 241.3     |
| Gradient argmax   | 141.86         | 549.0     |
| Expert 1 vs Expert 2 (baseline) | 1.38 | 5.3 |

The graph approach reaches inter-annotator agreement (~1.5 px ≈ 5.8 µm) and
generalises to a second subject (`Subject_06`: MASD ≈ 1.28 px).

---

## Repository layout

```
git_repo_v/
├── proj_notebook.ipynb     # main notebook (analysis, figures, evaluation)
├── proj_notebook.html      # rendered HTML export
├── src/                    # project modules
│   ├── preprocessing.py    # median + bilateral + CLAHE pipeline
│   ├── segmentation.py     # Dijkstra, cost image, ILM prior, multilayer
│   ├── baseline.py         # Otsu and gradient-argmax baselines
│   ├── evaluation.py       # Dice, MASD, max surface distance
│   └── extract_slices.py   # exports annotated slices from the .mat volume
├── tools/                  # course plotting/IO helpers
├── data/
│   ├── raw/2015_BOE_Chiu/  # Duke .mat volumes (not redistributed)
│   ├── slices/             # extracted Subject_02 B-scans (PNG)
│   ├── slices_sub06/       # extracted Subject_06 B-scans (PNG)
│   ├── preprocessed/       # denoised + CLAHE B-scans
│   └── ground_truth/       # expert ILM boundaries (.npz)
├── masks/
│   ├── auto/               # ±2 px masks from the Dijkstra segmentation
│   └── manual/             # ±2 px masks from the expert annotations
└── figures/                # generated figures used in the notebook
```

---

## Method

1. **Preprocessing** (`src/preprocessing.py`): 3×3 median filter →
   bilateral filter → CLAHE (8×8 tiles).
2. **Cost image** (`src/segmentation.py`): inverted, normalised vertical
   Sobel gradient; low cost on dark-above / bright-below transitions
   (the ILM signature). A mirrored cost handles bright-to-dark boundaries
   (RPE/choroid).
3. **Column prior**: per-column smoothed "first strong dark→bright
   transition" with MAD-based outlier rejection. Dijkstra is constrained
   to a ±25 px band around the prior to lock onto the ILM rather than
   deeper retinal edges or the cropped top band.
4. **Dijkstra**: 3-neighbour column graph (`(r,c) → (r±1 or r, c+1)`),
   `heapq` binary heap, `O(HW log HW)`.
5. **Multilayer**: sequential Dijkstra with exclusion bands below the
   previous boundary (ILM → IS/OS → RPE).

---

## Dataset

**Duke SD-OCT DME dataset** (Chiu et al., 2015):
<http://www.duke.edu/~sf59/Datasets/2015_BOE_Chiu2.zip>

- 10 subjects × 61 B-scans (496 × 768 px, 8-bit), Heidelberg Spectralis
  HRA+OCT, 3.87 µm/px axial.
- Two expert annotations of eight retinal layer boundaries on ~11 slices
  per subject.
- This project uses `Subject_02` (primary) and `Subject_06`
  (generalisation check).

License: research/educational use only; cite Chiu et al. (2015). The
`.mat` volumes are **not** redistributed here.

---

## Usage

Install dependencies:

```bash
pip install numpy scipy scikit-image opencv-python matplotlib pandas jupyter
```

Place the Duke `.mat` volumes under `data/raw/2015_BOE_Chiu/`, then:

```bash
# extract annotated slices + ground-truth ILM boundaries
python src/extract_slices.py

# run the full analysis
jupyter notebook proj_notebook.ipynb
```

The notebook regenerates every figure under `figures/` and writes
auto/manual masks under `masks/`.

---

Full list of references can be found in `proj_notebook.ipynb`