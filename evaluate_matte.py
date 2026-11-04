import argparse
import os
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np
from tqdm import tqdm

from utils import metric_GRAD, metric_MAD, metric_MSE


def main():
    parser = argparse.ArgumentParser(description="Compare predicted hair mattes to target mattes")
    parser.add_argument("--predictions-path", required=True, type=str, help="Folder of predicted mattes")
    parser.add_argument("--targets-path", required=True, type=str, help="Folder of target mattes")
    args = parser.parse_args()

    names = sorted(os.listdir(args.predictions_path))
    rows = []
    for name in tqdm(names, desc="Evaluating"):
        pred = cv2.imread(os.path.join(args.predictions_path, name), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(os.path.join(args.targets_path, name), cv2.IMREAD_GRAYSCALE)
        if pred.shape != target.shape:
            pred = cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_AREA)
        rows.append({
            "MAD": metric_MAD(pred, target),
            "MSE": metric_MSE(pred, target),
            "GRAD": metric_GRAD(pred, target),
        })

    print(f"images\t{len(rows)}")
    for key in ("MAD", "MSE", "GRAD"):
        print(f"{key}\t{float(np.mean([row[key] for row in rows])):.6f}")


if __name__ == "__main__":
    main()
