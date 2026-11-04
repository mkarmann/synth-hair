import json
from abc import ABC
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from torchvision import datasets
import yaml


# ---------
# Utilities
# ---------
def match_foreground_to_background(foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
    """
    Matches the foreground's RGB colors to the background using histogram matching.

    Args:
        foreground: (H1, W1, 4) uint8 (RGBA)
        background: (H2, W2, 3) uint8 (RGB)

    Returns:
        new_foreground: (H1, W1, 4) uint8
    """
    from skimage.exposure import match_histograms
    if foreground.shape[2] != 4:
        raise ValueError("Foreground must have 4 channels (RGBA)")
    if background.shape[2] != 3:
        raise ValueError("Background must have 3 channels (RGB)")

    # Split foreground into RGB and alpha
    alpha = foreground[:, :, 3]
    fg_rgb = foreground[..., :3][alpha > 0]

    # Match histogram of fg to bg (channel-wise)
    matched_rgb = match_histograms(fg_rgb, background.reshape((-1, 3)), channel_axis=-1)

    # Recombine with alpha
    new_foreground = foreground.copy()
    new_foreground[:, :, :3][alpha > 0] = matched_rgb
    return new_foreground

def compose_foreground_on_background(foreground: np.ndarray, background: np.ndarray):
    """

    Args:
        foreground: (H1, W1, 4) uint8
        background: (H2, W2, 3) uint8
    Returns:
        composed: (H1, W1, 3) uint8
    """
    H1, W1 = foreground.shape[:2]
    H2, W2 = background.shape[:2]

    # --- Step 1: Scale background to fit ---
    scale = max(H1 / H2, W1 / W2)  # ensures background fully covers foreground
    new_size = (int(round(W2 * scale)), int(round(H2 * scale)))
    bg_resized = cv2.resize(background, new_size, interpolation=cv2.INTER_LINEAR)

    # --- Step 2: Crop background to match (H1, W1) ---
    start_y = (bg_resized.shape[0] - H1) // 2
    start_x = (bg_resized.shape[1] - W1) // 2
    bg_cropped = bg_resized[start_y:start_y + H1, start_x:start_x + W1]

    # --- Step 3: Split foreground into color + alpha ---
    fg_rgb = foreground[..., :3].astype(np.float32)
    alpha = foreground[..., 3:4].astype(np.float32) / 255.0  # normalize to [0,1]

    # --- Step 4: Alpha blend ---
    composed = fg_rgb * alpha + bg_cropped.astype(np.float32) * (1 - alpha)
    composed = np.clip(composed, 0, 255).astype(np.uint8)

    return composed


_paths_flat: dict[str, str] = {}


def load_paths_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _flatten_paths_tree(node: dict, prefix: str = "") -> dict[str, str]:
    """Turn nested paths YAML into 'group/sub/key' -> absolute path."""
    out: dict[str, str] = {}
    for k, v in node.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_paths_tree(v, key))
        elif v is not None:
            out[key] = str(v).rstrip("/")
    return out


def load_paths_from_config(cfg: dict) -> None:
    """
    Load all dataset paths from a YAML dict (typically configs/paths.yaml).

    Requires a top-level ``paths:`` mapping (nested) with absolute paths for
    every dataset location the code uses. There is no implicit folder layout.
    """
    global _paths_flat
    paths = cfg.get("paths")
    if not isinstance(paths, dict) or not paths:
        raise ValueError(
            "Configuration must contain a non-empty top-level 'paths:' block "
            "(see configs/paths.yaml)."
        )
    _paths_flat = _flatten_paths_tree(paths)


def apply_paths_yaml(path: str | Path) -> None:
    """Load configs/paths.yaml (or equivalent) into the global path registry."""
    load_paths_from_config(load_paths_yaml(path))


def dataset_path(*segments: str) -> str:
    """
    Look up an absolute directory (or file root) from ``paths`` in configs/paths.yaml.

    Keys are built from the YAML tree, e.g. ``dataset_path('portrait_matting_datasets', 'Figaro1K')``
    matches ``paths.portrait_matting_datasets.Figaro1K`` in YAML.
    """
    key = "/".join(segments)
    if not _paths_flat:
        raise RuntimeError(
            "Dataset paths are not loaded. Load configs/paths.yaml before using datasets, e.g. "
            "run `python train.py --config <preset>` from the repo root (loads configs/paths.yaml automatically) or call "
            "`from datasets import apply_paths_yaml; apply_paths_yaml('configs/paths.yaml')`."
        )
    if key not in _paths_flat:
        preview = sorted(_paths_flat.keys())[:25]
        more = " ..." if len(_paths_flat) > 25 else ""
        raise KeyError(
            f"No path registered for {key!r}. Add it under `paths` in configs/paths.yaml. "
            f"Example keys: {preview}{more}"
        )
    return _paths_flat[key]


def synthetic_hair_root_for_resolution(training_resolution: int) -> str:
    return dataset_path("synthetic_hair", f"{training_resolution}x{training_resolution}")


# --------
# Datasets
# --------

class BGMv2BackgroundsDataset(torch.utils.data.Dataset):
    """
    Project: https://github.com/PeterL1n/BackgroundMattingV2?tab=readme-ov-file#download
    Download: https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets

    """
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = dataset_path("portrait_matting_datasets", "Backgrounds")
        assert os.path.exists(data_path)
        self.name = 'BGMv2 Backgrounds'
        self.data_path = data_path
        self.sample_ids = list(sorted([os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(data_path, '*.jpg'))]))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        image_path = os.path.join(self.data_path, f'{sample_id}.jpg')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        return {
            'image': image,  # (H, W, 3)
        }


class Figaro1kDataset(torch.utils.data.Dataset):
    """ https://www.michelesvanera.org/figaro-1k """
    def __init__(self, data_path=None, split='train'):
        if data_path is None:
            data_path = dataset_path("portrait_matting_datasets", "Figaro1K")
        assert split in ['train', 'test']
        assert os.path.exists(data_path)

        self.name = 'Figaro1k'
        self.split = split
        self.data_path = data_path
        self.split_folder = 'Training' if split == 'train' else 'Testing'
        self.image_folder = os.path.join(data_path, 'Figaro1k', 'Figaro1k', 'Original', self.split_folder)
        self.gt_folder = os.path.join(data_path, 'Figaro1k', 'Figaro1k', 'GT', self.split_folder)
        self.sample_ids = list(sorted([fname.split('-')[0] for fname in os.listdir(self.image_folder)]))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, item):
        sample_id = self.sample_ids[item]
        image_path = os.path.join(self.image_folder, f'{sample_id}-org.jpg')
        segmentation_path = os.path.join(self.gt_folder, f'{sample_id}-gt.pbm')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        segmentation = (cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED) >= 128).astype(int)
        return {
            'image': image,  # (H, W, 3)
            'segmentation': segmentation  # (H, W)
        }


class CelebAMaskHQDataset(torch.utils.data.Dataset):
    """
    https://github.com/switchablenorms/CelebAMask-HQ
    """
    def __init__(self, data_path=None, load_just_hair=True, load_just_scalp_hair=False):
        if data_path is None:
            data_path = dataset_path("portrait_matting_datasets", "CelebAMask-HQ")
        assert load_just_hair or load_just_scalp_hair
        assert not (load_just_scalp_hair and load_just_hair)
        assert os.path.exists(data_path)

        self.load_just_scalp_hair = load_just_scalp_hair

        self.name = 'CelebAMask-HQ'
        self.data_path = data_path
        self.images_folder = os.path.join(data_path, 'CelebA-HQ-img')
        self.labels_folder = os.path.join(data_path, 'CelebAMask-HQ-mask-anno')
        self.sample_ids = list(sorted([int(f.split('.')[0]) for f in os.listdir(self.images_folder)]))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        image_path = os.path.join(self.images_folder, f'{sample_id}.jpg')
        label_hair_path = os.path.join(self.labels_folder, str(sample_id // 2000), f'{sample_id:05d}_hair.png')
        label_l_brow_path = os.path.join(self.labels_folder, str(sample_id // 2000), f'{sample_id:05d}_l_brow.png')
        label_r_brow_path = os.path.join(self.labels_folder, str(sample_id // 2000), f'{sample_id:05d}_r_brow.png')

        assert os.path.exists(image_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        segmentation = np.zeros((512, 512), dtype=int)
        if os.path.exists(label_hair_path):
            label_hair = (cv2.imread(label_hair_path, cv2.IMREAD_GRAYSCALE) > 127).astype(int)
            segmentation += label_hair
        if not self.load_just_scalp_hair:
            if os.path.exists(label_l_brow_path):
                label_l_brow = (cv2.imread(label_l_brow_path, cv2.IMREAD_GRAYSCALE) > 127).astype(int)
                segmentation += label_l_brow
            if os.path.exists(label_r_brow_path):
                label_r_brow = (cv2.imread(label_r_brow_path, cv2.IMREAD_GRAYSCALE) > 127).astype(int)
                segmentation += label_r_brow
        segmentation = (segmentation > 0).astype(int)

        return {
            'image': image,  # (H, W, 3)
            'segmentation': segmentation
        }


class HiSa(Dataset):
    """
    https://drive.google.com/drive/folders/16_wS405x3sTCbIo2rPXr9q6Y_kuS6tk-
    From the HairStep Paper.
    HiSa is the Strand map dataset vs. HiDA, the depth version
    """
    def __init__(self,
                 data_path=None,
                 split: str = 'train'):
        if data_path is None:
            data_path = dataset_path("portrait_matting_datasets", "HiSa_HiDa")
        assert split in ['train', 'test']
        assert os.path.exists(data_path)

        self.name = 'HiSa'
        self.split = split
        self.matte_folder = data_path
        with open(os.path.join(data_path, f'split_{split}.json'), 'r') as f:
            self.sample_ids = json.load(f)
        self.image_folder = os.path.join(data_path, 'img')
        self.matte_folder = os.path.join(data_path, 'seg')
        self.strand_map_folder = os.path.join(data_path, 'strand_map')

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, item):
        sample_id = self.sample_ids[item]
        image_path = os.path.join(self.image_folder, f'{sample_id}')
        matte_path = os.path.join(self.matte_folder, f'{sample_id}')
        strand_map_path = os.path.join(self.strand_map_folder, f'{sample_id}')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        matte = cv2.imread(matte_path, cv2.IMREAD_GRAYSCALE)
        strand_map = cv2.imread(strand_map_path, cv2.IMREAD_COLOR_RGB)
        assert image is not None
        assert matte is not None
        assert strand_map is not None

        return {
            'image': image,  # (H, W, 3)
            'matte': matte,  # (H, W)
            'strand_map': strand_map    # (H, W, 3) R-channel: (0: background, 128: person, 255: hair), G and B are direction
        }


class FaceSyntheticsDataset(Dataset):
    """
    Face Synthetics dataset from Microsoft
    Used in paper: Fake it till you make it: face analysis in the wild using only synthetic data only

    Download from GitHub: https://github.com/microsoft/FaceSynthetics?tab=readme-ov-file

    Segmentation labels:
        BACKGROUND = 0
        SKIN = 1
        NOSE = 2
        RIGHT_EYE = 3
        LEFT_EYE = 4
        RIGHT_BROW = 5
        LEFT_BROW = 6
        RIGHT_EAR = 7
        LEFT_EAR = 8
        MOUTH_INTERIOR = 9
        TOP_LIP = 10
        BOTTOM_LIP = 11
        NECK = 12
        HAIR = 13
        BEARD = 14
        CLOTHING = 15
        GLASSES = 16
        HEADWEAR = 17
        FACEWEAR = 18
        IGNORE = 255

    """
    def __init__(self,
                 data_path=None):
        if data_path is None:
            data_path = dataset_path("portrait_matting_datasets", "FaceSynthetics")
        assert os.path.isdir(data_path)

        self.name = 'FaceSynthetics'
        self.data_path = data_path
        self.sample_ids = list(f'{i:06d}' for i in range(100_000))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, item):
        sample_id = self.sample_ids[item]

        image_path = os.path.join(self.data_path, f'{sample_id}.png')
        segmentation_path = os.path.join(self.data_path, f'{sample_id}_seg.png')

        assert os.path.isfile(image_path)
        assert os.path.isfile(segmentation_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        segmentation = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)

        return {
            'image': image,  # (512, 512, 3)
            'segmentation': segmentation,  # (512, 512)
        }


class ClothingDataset(Dataset):
    """
    https://github.com/alexeygrigorev/clothing-dataset

    Downloaded from kaggle: https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full?resource=download
    """
    def __init__(self, data_path=None, load_original_images=False):
        if data_path is None:
            data_path = dataset_path("portrait_matting_datasets", "Clothing")
        assert os.path.isdir(data_path)

        self.name = 'Clothing Dataset'
        self.data_path = data_path
        self.df = pd.read_csv(os.path.join(data_path, 'images.csv'), index_col='image')
        self.sample_ids = list(self.df.index)
        self.images_folder = os.path.join(self.data_path, 'images_original' if load_original_images else 'images_compressed')

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, item):
        sample_id = self.sample_ids[item]

        image_path = os.path.join(self.images_folder, f'{sample_id}.jpg')
        assert os.path.isfile(image_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)

        return {
            'image': image,  # (H, W, 3)
        }


class SynthHairAugmentations:
    def __init__(self):
        self.augmentation_pipeline = self.create_augmentation_function()

    def create_augmentation_function(self):
        import albumentations as A

        augment_motion_blur = A.Compose([
            A.MotionBlur(blur_limit=5, angle_range=(0, 360), direction_range=(0.0, 0.0), p=0.15),
            A.Defocus(radius=5, alias_blur=(0.1, 0.5), p=0.15),
        ], additional_targets={
            'matte': 'image',
            'scalp_hair_matte': 'image',
            'eyebrows_matte': 'image',
            'eyelashes_matte': 'image',
            'mustache_matte': 'image',
            'beard_matte': 'image',
            'hair_growing_direction': 'mask',
            'depth': 'mask'
        })

        augment_color_noise = A.Compose([
            A.ColorJitter(brightness=0.3, contrast=(0.8, 1.4), saturation=0.3, hue=0.05, p=0.7),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.ToGray(p=0.1),
            A.ImageCompression(quality_range=(60, 90), p=0.3),
        ], additional_targets={
            'matte': 'mask',
            'scalp_hair_matte': 'mask',
            'eyebrows_matte': 'mask',
            'eyelashes_matte': 'mask',
            'mustache_matte': 'mask',
            'beard_matte': 'mask',
            'hair_growing_direction': 'mask',
            'depth': 'mask'
        })

        augmentation_pipeline = A.Compose([
            augment_motion_blur,
            augment_color_noise,
        ])

        return augmentation_pipeline

    @staticmethod
    def horizontal_flip_augmentation(data: dict):
        if random.random() < 0.5:
            # Flip
            for k, v in data.items():
                v = v[:, ::-1]

                # We have to also flip the coordinate system of our hair growing direction
                if k == 'hair_growing_direction':
                    x_channel = v[:, :, 1]
                    mask = v[:, :, 3] != 0
                    x_channel[mask] = 255 - x_channel[mask]
                data[k] = v.copy()
        return data

    @staticmethod
    def crop_augmentation(data: dict):
        if random.random() < 0.1:
            # Crop size
            bar_size = random.uniform(0.0, 0.2)
            if random.random() < 0.5:
                for k, v in data.items():
                    H, W = v.shape[:2]
                    a = int(round(bar_size * W))
                    b = int(round(W - bar_size * W))
                    v = v.copy()
                    v[:, :a] = 0
                    v[:, b:] = 0
                    data[k] = v
            else:
                for k, v in data.items():
                    H, W = v.shape[:2]
                    a = int(round(bar_size * H))
                    b = int(round(H - bar_size * H))
                    v = v.copy()
                    v[:a] = 0
                    v[b:] = 0
                    data[k] = v
        return data

    def __call__(self, sample):
        sample = self.augmentation_pipeline(**sample)
        sample = SynthHairAugmentations.horizontal_flip_augmentation(sample)
        sample = SynthHairAugmentations.crop_augmentation(sample)
        return sample


class SynthHairDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 split='train',
                 resize=None,
                 transform=None,
                 load_only_hair_matte=False,
                 load_only_hair_matte_with_categories=False,
                 use_train_augmentation=False,
                 disable_background=False):
        if data_path is None:
            data_path = dataset_path("synthetic_hair", "512x512")
        assert os.path.isdir(data_path)
        assert split in ['train', 'test']

        self.name = 'SynthHair'
        self.split = split
        self.load_only_hair_matte = load_only_hair_matte
        self.load_only_hair_matte_with_categories = load_only_hair_matte_with_categories
        self.data_path = data_path
        self.background_datasets = None if disable_background else [BGMv2BackgroundsDataset(), ClothingDataset(load_original_images=True)]
        self.resize = resize
        self.samples = pd.read_csv(os.path.join(data_path, 'samples.csv'), index_col='id')
        self.samples = self.samples[self.samples['split'] == split]
        self.sample_ids = list(self.samples.index)
        self.transform = transform
        self.is_train = use_train_augmentation
        self.augmentation_pipeline = SynthHairAugmentations() if use_train_augmentation else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample_id = self.sample_ids[item]
        row = self.samples.loc[sample_id]

        def _path(column):
            path = os.path.join(self.data_path, row[column])
            assert os.path.isfile(path), path
            return path

        image_path = _path('image')
        hair_matte_path = _path('hair_matte')

        # Load image and draw a white background
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        portrait_matte_float = image[:, :, 3] / 255
        hair_matte = cv2.imread(hair_matte_path, cv2.IMREAD_GRAYSCALE)

        if self.load_only_hair_matte:
            scalp_hair_matte = eyebrows_matte = eyelashes_matte = mustache_matte = beard_matte = None
            hair_growing_direction = None
            depth = None
        else:
            scalp_hair_matte = cv2.imread(_path('scalp_hair_matte'), cv2.IMREAD_GRAYSCALE)
            eyebrows_matte = cv2.imread(_path('eyebrows_matte'), cv2.IMREAD_GRAYSCALE)
            eyelashes_matte = cv2.imread(_path('eyelashes_matte'), cv2.IMREAD_GRAYSCALE)
            mustache_matte = cv2.imread(_path('mustache_matte'), cv2.IMREAD_GRAYSCALE)
            beard_matte = cv2.imread(_path('beard_matte'), cv2.IMREAD_GRAYSCALE)
            if self.load_only_hair_matte_with_categories:
                hair_growing_direction = None
                depth = None
            else:
                hair_growing_direction = cv2.cvtColor(cv2.imread(_path('hair_growing_direction'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
                depth = cv2.imread(_path('depth'), cv2.IMREAD_UNCHANGED)

        # Add background to image
        if self.background_datasets is None:
            # Light Gray background
            image = (portrait_matte_float[:, :, None] * image[:, :, :3] + (1 - portrait_matte_float[:, :, None]) * 255).astype(np.uint8)
        else:
            bg_dataset = random.choice(self.background_datasets) if self.is_train else self.background_datasets[0]
            bg_idx = random.randint(0, len(bg_dataset) - 1) if self.is_train else item % len(bg_dataset)
            bg = bg_dataset[bg_idx]['image']
            if self.is_train:
                bg = SynthHairAugmentations.horizontal_flip_augmentation({'image': bg})['image']
            image_matched = match_foreground_to_background(image, bg)
            matching_mix = random.uniform(0.2, 0.35) if self.is_train else 0.3
            image = ((1 - matching_mix) * image + matching_mix * image_matched).astype(np.uint8)
            image = compose_foreground_on_background(image, bg)

        if self.load_only_hair_matte:
            out = {
                'image': image, # (H, W, 3) dtype: np.uint8
                'matte': hair_matte, # (H, W) dtype: np.uint8
            }
        elif self.load_only_hair_matte_with_categories:
            out = {
                'image': image,  # (H, W, 3) dtype: np.uint8
                'matte': hair_matte,  # (H, W) dtype: np.uint8
                'scalp_hair_matte': scalp_hair_matte,  # (H, W) dtype: np.uint8
                'eyebrows_matte': eyebrows_matte,  # (H, W) dtype: np.uint8
                'eyelashes_matte': eyelashes_matte,  # (H, W) dtype: np.uint8
                'mustache_matte': mustache_matte,  # (H, W) dtype: np.uint8
                'beard_matte': beard_matte,  # (H, W) dtype: np.uint8
            }
        else:
            out = {
                'image': image,  # (H, W, 3) dtype: np.uint8
                'matte': hair_matte,  # (H, W) dtype: np.uint8
                'scalp_hair_matte': scalp_hair_matte,  # (H, W) dtype: np.uint8
                'eyebrows_matte': eyebrows_matte,  # (H, W) dtype: np.uint8
                'eyelashes_matte': eyelashes_matte,  # (H, W) dtype: np.uint8
                'mustache_matte': mustache_matte,  # (H, W) dtype: np.uint8
                'beard_matte': beard_matte,  # (H, W) dtype: np.uint8
                'hair_growing_direction': hair_growing_direction,  # (H, W, 4) dtype: np.uint8
                'depth': depth,  # (H, W) dtype: float
            }

        if self.is_train:
            out = self.augmentation_pipeline(out)

        if self.transform is not None:
            for k in out.keys():
                out[k] = self.transform(out[k])

        return out

    def plot_sample(self, idx):
        sample = self[idx]
        print('Video frame:')
        print(self.samples['video_frame'][self.sample_ids[idx]])
        n_keys = len(sample)
        fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True, sharey=True)
        if n_keys == 1:
            axes = [axes]
        for ax, (key, value) in zip(axes.flatten(), sample.items()):
            # Handle grayscale, RGB, RGBA
            if value.ndim == 2:
                if key == 'depth':
                    mask = value != 0
                    value[mask] = 1 / value[mask]
                    ax.imshow(value, cmap='gnuplot2', vmin=np.min(value[mask]), vmax=np.max(value[mask]))
                else:
                    ax.imshow(value, cmap='gray')
            elif value.ndim == 3 and value.shape[2] == 3:
                ax.imshow(value)
            elif value.ndim == 3 and value.shape[2] == 4:
                # RGBA
                if key == 'depth':
                    value = value[:, :, 0]
                    mask = value != 0
                    value[mask] = 1 / value[mask]
                    ax.imshow(value, cmap='gnuplot2', vmin=np.min(value[mask]), vmax=np.max(value[mask]))
                elif key == 'hair_growing_direction':
                    # Extract 2D directions (dx, dy)
                    directions_2d = value[:, :, [1, 2]]

                    # Mask valid pixels (where 4th channel > 0)
                    mask = value[:, :, 3] > 0
                    valid_dirs = directions_2d[mask]

                    # Get coordinates of valid pixels
                    coords = np.column_stack(np.where(mask))

                    # Randomly choose up to 1000 valid pixels
                    num_samples = min(1000, len(coords))
                    arrow_idx = np.random.choice(len(coords), num_samples, replace=False)

                    sample_coords = coords[arrow_idx]
                    sample_dirs = valid_dirs[arrow_idx] / 255 - 0.5

                    # Plot image
                    ax.imshow(value)

                    # Plot arrows
                    for (y, x), (dx, dy) in zip(sample_coords, sample_dirs):
                        ax.arrow(
                            x,
                            y,
                            dx * 50,
                            -dy * 50,
                            head_width=5,
                            head_length=10,
                            length_includes_head=True,
                            linewidth=0.5,  # thicker line
                            color='black',  # black line + black head fill
                            fc='black'  # explicitly ensure head fill is black
                        )
                else:
                    ax.imshow(value)
            else:
                ax.imshow(value)
            ax.set_title(f"{key}")
            ax.axis('off')
        split_str = f" ({self.split})"
        fig.suptitle(f"{self.name}{split_str} - Sample id {self.sample_ids[idx]} at index {idx}")
        plt.tight_layout()
        plt.show()


