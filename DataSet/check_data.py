from pathlib import Path
import numpy as np

data = np.load(Path(__file__).resolve().parent / "dataset.npz", allow_pickle=False)
for split in ["train", "test"]:
    x, y = data["x_" + split], data["y_" + split]
    print(split, "images", x.shape, x.dtype, "labels", y.shape, y.dtype)
    print("pixel range", int(x.min()), int(x.max()))
    print("class counts", dict(zip(*np.unique(y, return_counts=True))))
# PyTorch inputs: x.astype(np.float32) / 255.0 in N,C,H,W order.
# Keras inputs: np.transpose(x, (0,2,3,1)).astype(np.float32) / 255.0.
