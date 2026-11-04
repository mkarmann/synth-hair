import argparse
import os
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
from PIL import Image

from soft_ipq import soft_IPQ

_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Soft-IPQ for each matte in a folder, then the average")
    parser.add_argument("--input-path", type=str, required=True, help="Directory containing matte images")
    args = parser.parse_args()

    folder = args.input_path
    names = sorted(n for n in os.listdir(folder) if Path(n).suffix.lower() in _EXT)
    if not names:
        print("No images found.")
        return

    scores = []
    for n in names:
        path = os.path.join(folder, n)
        arr = np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0
        s = soft_IPQ(arr)
        scores.append(s)
        print(f"{path}\t{s:.6f}")

    print(f"average_sipq\t{float(np.mean(scores)):.6f}")


if __name__ == "__main__":
    main()
