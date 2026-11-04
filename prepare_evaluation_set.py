import argparse
import json
import os
import random
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np
import tifffile
from torch.utils.data import Subset
from tqdm import tqdm

from datasets import (
    CelebAMaskHQDataset,
    FaceSyntheticsDataset,
    Figaro1kDataset,
    HiSa,
    SynthHairDataset,
    apply_paths_yaml,
    synthetic_hair_root_for_resolution,
)
from utils import RemapHairCategoryMattesToSegmentationTargets, RemapHairGrowingDirectionTo2DOrientation


class RemapFaceSyntheticsToCategorySegmentation:
    def __init__(self, dataset, category):
        assert category in ("scalp_hair", "eyebrows", "mustache_beard")
        self.dataset = dataset
        self.category = category

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        seg = sample["segmentation"]
        if self.category == "scalp_hair":
            binary = (seg == 13).astype(np.int32)
        elif self.category == "eyebrows":
            binary = ((seg == 5) | (seg == 6)).astype(np.int32)
        else:
            binary = (seg == 14).astype(np.int32)
        return {"image": sample["image"], "segmentation": binary}


def _shuffled_ids(n_total, n, seed):
    random.seed(int(seed))
    ids = list(range(n_total))
    random.shuffle(ids)
    return ids[:n]


def prepare_dataset(dataset, out_folder, hair_categories_in_labels, source_ids=None):
    os.makedirs(out_folder, exist_ok=True)
    with open(os.path.join(out_folder, "hair_categories.json"), "w") as f:
        json.dump(hair_categories_in_labels, f, indent=4)
    if source_ids is not None:
        with open(os.path.join(out_folder, "ids.json"), "w") as f:
            json.dump(source_ids, f, indent=4)
    for i, sample in enumerate(tqdm(dataset, "Preparing dataset")):
        sample_fname = f"{i}.png"
        if "image" in sample:
            folder = os.path.join(out_folder, "images")
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, sample_fname), sample["image"][:, :, ::-1])
        if "matte" in sample:
            folder = os.path.join(out_folder, "labels", "matte")
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, sample_fname), sample["matte"])
        if "segmentation" in sample:
            folder = os.path.join(out_folder, "labels", "segmentation")
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, sample_fname), (sample["segmentation"] * 255).astype(np.uint8))
        if "hair_category" in sample:
            folder = os.path.join(out_folder, "labels", "hair_category")
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, sample_fname), sample["hair_category"].astype(np.uint8))
        if "hair_category_mask" in sample:
            folder = os.path.join(out_folder, "labels", "hair_category_mask")
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, sample_fname), (sample["hair_category_mask"] * 255).astype(np.uint8))
        if "2d_orientation" in sample:
            folder = os.path.join(out_folder, "labels", "2d_orientation")
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, sample_fname), sample["2d_orientation"][:, :, ::-1])
        if "strand_map" in sample:
            folder = os.path.join(out_folder, "labels", "strand_map")
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, sample_fname), sample["strand_map"][:, :, ::-1])
        if "depth" in sample:
            folder = os.path.join(out_folder, "labels", "depth")
            os.makedirs(folder, exist_ok=True)
            tifffile.imwrite(os.path.join(folder, f"{i}.tiff"), sample["depth"], compression="lzw")


def _synth_hair(resolution):
    return RemapHairGrowingDirectionTo2DOrientation(RemapHairCategoryMattesToSegmentationTargets(
        SynthHairDataset(synthetic_hair_root_for_resolution(resolution), split="test", use_train_augmentation=False)
    ))


def prepare_dataset_by_name(out_folder, dataset, seed):
    out_folder = os.path.join(out_folder, dataset)
    print("Preparing:", dataset, "in", out_folder)
    if dataset == "SynthHairx512":
        prepare_dataset(_synth_hair(512), out_folder, [0, 1, 2, 3, 4])
    elif dataset == "SynthHairx1024":
        prepare_dataset(_synth_hair(1024), out_folder, [0, 1, 2, 3, 4])
    elif dataset == "SynthHairx2048":
        prepare_dataset(_synth_hair(2048), out_folder, [0, 1, 2, 3, 4])
    elif dataset == "SynthHairx4096":
        prepare_dataset(_synth_hair(4096), out_folder, [0, 1, 2, 3, 4])
    elif dataset == "Figaro1k":
        prepare_dataset(Figaro1kDataset(split="test"), out_folder, [0])
    elif dataset == "HiSa":
        prepare_dataset(HiSa(split="test"), out_folder, [0])
    elif dataset == "CelebAMaskHQ":
        ids = _shuffled_ids(len(CelebAMaskHQDataset()), 10000, seed)
        prepare_dataset(Subset(CelebAMaskHQDataset(), ids), out_folder, [0, 1])
    elif dataset == "CelebAMaskHQScalpHair":
        dataset_obj = CelebAMaskHQDataset(load_just_hair=False, load_just_scalp_hair=True)
        ids = _shuffled_ids(len(dataset_obj), 10000, seed)
        prepare_dataset(Subset(dataset_obj, ids), out_folder, [0])
    elif dataset == "FaceSyntheticsEyebrows-300":
        ids = _shuffled_ids(len(FaceSyntheticsDataset()), 300, seed)
        prepare_dataset(
            RemapFaceSyntheticsToCategorySegmentation(Subset(FaceSyntheticsDataset(), ids), "eyebrows"),
            out_folder,
            [1],
            source_ids=ids,
        )
    elif dataset == "FaceSyntheticsMustacheBeard-300":
        ids = _shuffled_ids(len(FaceSyntheticsDataset()), 300, seed)
        prepare_dataset(
            RemapFaceSyntheticsToCategorySegmentation(Subset(FaceSyntheticsDataset(), ids), "mustache_beard"),
            out_folder,
            [3, 4],
            source_ids=ids,
        )
    elif dataset == "FaceSyntheticsScalpHair-300":
        ids = _shuffled_ids(len(FaceSyntheticsDataset()), 300, seed)
        prepare_dataset(
            RemapFaceSyntheticsToCategorySegmentation(Subset(FaceSyntheticsDataset(), ids), "scalp_hair"),
            out_folder,
            [0],
            source_ids=ids,
        )
    else:
        raise SystemExit(f'Could not find dataset "{dataset}"')


def main():
    parser = argparse.ArgumentParser(description="Export a test split into an evaluation folder")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name, for example SynthHairx4096")
    parser.add_argument("--output-path", default="evaluation", type=str, help="Directory that will contain the dataset folder")
    parser.add_argument("--seed", default=2025, type=int, help="Shuffle seed for subsampled sets")
    args = parser.parse_args()

    apply_paths_yaml(Path(__file__).resolve().parent / "configs" / "paths.yaml")
    prepare_dataset_by_name(args.output_path, args.dataset, args.seed)


if __name__ == "__main__":
    main()
