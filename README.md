# CIFAR-10 Three-Class Vehicle Classification

A small CNN trained **from scratch** (PyTorch) to classify 32×32 colour images into **airplane, automobile, ship**
(a 3-class subset of CIFAR-10). The project compares training **without augmentation** (experiment A) against training
with **random horizontal flip on the training set only** (experiment B). Everything else is identical.

## Folder structure

```
CIFAR10_Three_Class_Vehicle_Classification/
├── README.md
├── requirements.txt
├── notebook/
│   └── notebook.ipynb  # the full pipeline (run this)
├── img/                # figures saved by the notebook (created if missing)
├── models/             # best-validation weights of each experiment (created if missing)
└── results/            # logs, tables, metrics (created if missing)
```

The local `DataSet/`, `report/`, and `models/` folders are excluded from Git. `DataSet/` must be present locally to run the notebook; it should contain the project dataset (`dataset.npz`) and label mapping (`labels.txt`). Model checkpoints are generated when the notebook runs. The report is a local deliverable and is not required to run the notebook.

## Setup and run

```powershell
pip install -r requirements.txt
jupyter notebook notebook/notebook.ipynb
```

Then run all cells (*Kernel → Restart & Run All*). The notebook works whether Jupyter is started from the project root or from `notebook/`.
It needs no internet connection. A normal laptop CPU is enough (the network is tiny, so a run should take a few minutes at most); a GPU is used automatically if PyTorch can see one.

`requirements.txt` installs the default PyTorch build from PyPI. For a specific CUDA build, install PyTorch first following https://pytorch.org/get-started/locally/ and then the remaining packages.

## What the notebook does

1. **Setup & reproducibility:** imports, then one parameters cell (seed, epochs, batch size, learning rate, paths, options). The parameters and software versions are saved to `results/run_config.json`.
2. **Data:** loads `dataset.npz`, checks shapes, dtypes, pixel range and class counts (same checks as `check_data.py`), verifies `labels.txt`, and saves a sample grid.
3. **Split:** per-class 1/5 of the training pool goes to validation (4,000 train + 1,000 validation per class), seed 42; indices saved to `results/split_indices.npz`. The test set stays separate.
4. **Preprocessing:** `float32 / 255`; labels untouched.
5. **Model:** `Conv(16)→ReLU→MaxPool → Conv(32)→ReLU→MaxPool → Flatten → Linear(64)→ReLU → Linear(3)`; raw logits go to `CrossEntropyLoss`. Per-layer shapes for one batch are printed.
6. **Training:** 5 epochs, batch size 64, learning rate 0.001, AdamW. Per-epoch train/validation loss and accuracy and time are logged; the best weights by validation accuracy are saved.
   - **A:** no augmentation.
   - **B:** random horizontal flip (p = 0.5) on training batches only, trained from scratch with the same split, initial weights and batch order.
7. **Test:** each model's best weights are evaluated on the test set once. Outputs: overall accuracy and loss, per-class accuracy, confusion matrices, misclassified images.
8. **Analysis:** border-colour statistics of correct vs misclassified images (to examine the coloured-background effect), an auto-generated summary with a paired McNemar test, and a written discussion.
9. **Optional:** `RUN_EXTRA_SEEDS = True` repeats A and B with additional seeds and reports mean ± std.

## Discussion and conclusions

The completed notebook discussion interprets the results using the generated metrics, confusion matrices, misclassified-image grids, and border-colour statistics.

For the recorded seed-42 run, the no-augmentation model (A) achieved 82.17% best validation accuracy and 82.63% test accuracy. The random-horizontal-flip model (B) achieved 82.87% validation accuracy and 83.20% test accuracy, an observed improvement of 0.57 percentage points. The paired McNemar test returned `p = 0.207`, so this single five-epoch run does not establish a statistically reliable advantage for augmentation.

Horizontal flipping improved airplane accuracy from 85.8% to 87.2% and ship accuracy from 70.9% to 72.7%, but automobile accuracy decreased from 91.2% to 89.7%. The ship class remained the weakest class. The most frequent confusion was ship predicted as airplane, consistent with both classes often appearing against blue sky or water backgrounds. Border-colour statistics also suggest that the model uses background colour as a shortcut. Horizontal flipping changes object orientation but not colour, so it is not expected to eliminate these background-driven errors.

At 32×32 resolution, details such as wings, hulls, masts, and wheels are difficult to resolve, while viewpoint, scale, occlusion, and clutter vary substantially. Stronger follow-up experiments could use multiple seeds, longer training, colour-aware augmentation, random crops, or a deeper network.

## Outputs

| Location | File | Content |
|---|---|---|
| `img/` | `sample_images.png` | random training samples per class |
| | `accuracy_curves.png`, `loss_curves.png` | train/validation curves for A and B |
| | `validation_accuracy_comparison.png` | validation accuracy, A vs B |
| | `confusion_matrices.png`, `confusion_matrix_<exp>.png` | test confusion matrices (rows = true, columns = predicted) |
| | `per_class_accuracy.png` | per-class test accuracy, A vs B |
| | `misclassified_<exp>.png` | misclassified test images (at least 6, or all if fewer) |
| `results/` | `run_config.json` | parameters and environment |
| | `split_indices.npz` | train/validation indices |
| | `<exp>_log.csv`, `<exp>_log.txt` | per-epoch training logs |
| | `comparison.csv`, `per_class_accuracy.csv`, `test_metrics.json` | final metrics |
| | `confusion_matrix_<exp>.csv`, `misclassified_<exp>.csv` | confusion matrices; full list of errors |
| | `background_color_stats.csv` | border colour of correct vs misclassified images |
| | `summary.txt` | generated summary of the comparison |
| `models/` | `<exp>_best.pt` | best-validation weights |

`<exp>` is `A_no_augmentation` or `B_hflip`.

## Reproducibility

- Seed 42 drives the split, weight initialisation, batch order and flip randomness. Deterministic PyTorch kernels are requested (`DETERMINISTIC = True`).
- A and B start from identical weights (the notebook prints a fingerprint to confirm it).
- The same machine and software versions give the same numbers; other hardware or PyTorch versions can cause tiny floating-point differences.
- Change settings only in the parameters cell near the top of the notebook.

## Notes

- The test set is never used for training, tuning or model selection; it is evaluated once per model.
- With a single run and 5 epochs, small differences between A and B may be noise. The notebook reports a paired McNemar test, and the optional multi-seed run measures run-to-run variation.
- Results are generated by running the notebook; read the numbers from `results/summary.txt` and `results/comparison.csv`.
