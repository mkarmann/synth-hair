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

from utils import metric_ACC, metric_F1, metric_mIoU


def main():
    parser = argparse.ArgumentParser(description="Compare predicted mattes to binary segmentation targets")
    parser.add_argument("--predictions-path", required=True, type=str, help="Folder of predicted mattes")
    parser.add_argument("--targets-path", required=True, type=str, help="Folder of target segmentations")
    parser.add_argument("--threshold", default=0.4, type=float, help="Matte value at or above which a pixel is hair")
    args = parser.parse_args()

    names = sorted(os.listdir(args.predictions_path))
    rows = []
    for name in tqdm(names, desc="Evaluating"):
        pred = cv2.imread(os.path.join(args.predictions_path, name), cv2.IMREAD_GRAYSCALE)
        target = (cv2.imread(os.path.join(args.targets_path, name), cv2.IMREAD_GRAYSCALE) >= 128).astype(np.uint8)
        if pred.shape != target.shape:
            pred = cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_AREA)
        pred = (pred / 255 >= args.threshold).astype(np.uint8)
        rows.append({
            "ACC": metric_ACC(pred, target),
            "mIoU": metric_mIoU(pred, target),
            "F1": metric_F1(pred, target),
        })

    print(f"images\t{len(rows)}")
    print(f"threshold\t{args.threshold}")
    for key in ("ACC", "mIoU", "F1"):
        print(f"{key}\t{float(np.mean([row[key] for row in rows])):.6f}")


if __name__ == "__main__":
    main()
