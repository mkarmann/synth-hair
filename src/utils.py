import os
import random
import shutil
from pathlib import Path

import cv2
import torchvision.transforms.functional as F_transforms
import math
import torch
import torch.nn.functional as F
from datetime import datetime
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import Dataset

from datasets import FaceSyntheticsDataset, SynthHairAugmentations


def set_all_seeds(seed: int = 42):
    # 1. Python built-in randomness
    random.seed(seed)

    # 2. Environment variable (Python hash seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 3. NumPy
    np.random.seed(seed)

    # 4. PyTorch (CPU)
    torch.manual_seed(seed)

    # 5. PyTorch (GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # 6. cuDNN (ensures determinism)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # (Optional but good practice for new PyTorch versions)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)


def _tiled_sum(pred: np.ndarray, target: np.ndarray, reduce, tile=512):
    height, width = pred.shape
    total = 0.0
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            y1, x1 = min(y + tile, height), min(x + tile, width)
            diff = pred[y:y1, x:x1].astype(np.float64) / 255.0 - target[y:y1, x:x1].astype(np.float64) / 255.0
            total += reduce(diff)
    return total / (height * width)


def metric_MAD(pred: np.ndarray, target: np.ndarray):
    return _tiled_sum(pred, target, lambda diff: np.abs(diff).sum())


def metric_MSE(pred: np.ndarray, target: np.ndarray):
    return _tiled_sum(pred, target, lambda diff: np.square(diff).sum())


def metric_GRAD(pred: np.ndarray, target: np.ndarray):
    # Same Sobel gradient error as convolving both mattes with
    # [[1, 0, -1], [2, 0, -2], [1, 0, -1]] and its transpose.
    # Tiles keep the 4K case from building four full-frame float64 maps.
    height, width = pred.shape
    grad_sum = 0.0
    tile = 512
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            y1, x1 = min(y + tile, height), min(x + tile, width)
            ys = np.arange(y - 1, y1 + 1)
            xs = np.arange(x - 1, x1 + 1)
            ys = np.where(ys < 0, -ys - 1, ys)
            ys = np.where(ys >= height, 2 * height - 1 - ys, ys)
            xs = np.where(xs < 0, -xs - 1, xs)
            xs = np.where(xs >= width, 2 * width - 1 - xs, xs)
            pred_w = pred[np.ix_(ys, xs)].astype(np.float64) / 255.0
            target_w = target[np.ix_(ys, xs)].astype(np.float64) / 255.0
            pred_dx = (pred_w[:-2, 2:] + 2 * pred_w[1:-1, 2:] + pred_w[2:, 2:]) - (pred_w[:-2, :-2] + 2 * pred_w[1:-1, :-2] + pred_w[2:, :-2])
            target_dx = (target_w[:-2, 2:] + 2 * target_w[1:-1, 2:] + target_w[2:, 2:]) - (target_w[:-2, :-2] + 2 * target_w[1:-1, :-2] + target_w[2:, :-2])
            pred_dy = (pred_w[2:, :-2] + 2 * pred_w[2:, 1:-1] + pred_w[2:, 2:]) - (pred_w[:-2, :-2] + 2 * pred_w[:-2, 1:-1] + pred_w[:-2, 2:])
            target_dy = (target_w[2:, :-2] + 2 * target_w[2:, 1:-1] + target_w[2:, 2:]) - (target_w[:-2, :-2] + 2 * target_w[:-2, 1:-1] + target_w[:-2, 2:])
            grad_sum += np.square(pred_dx - target_dx).sum() + np.square(pred_dy - target_dy).sum()
    return grad_sum / (height * width)


def metric_ACC(pred: np.ndarray, target: np.ndarray):
    """

    Args:
        pred: (H, W): int
        target: (H, W): int
        threshold:

    Returns:

    """
    assert len(pred.shape) == 2
    assert len(target.shape) == 2
    assert np.max(pred) <= 1
    assert np.max(target) <= 1

    pred_hard = pred.astype(int)
    target = target.astype(int)
    comparison = 1 - np.abs(target - pred_hard)
    return np.mean(comparison)


def metric_mIoU(pred: np.ndarray, target: np.ndarray):
    """

    Args:
        pred: (H, W): int
        target: (H, W): int

    Returns:

    """
    assert len(pred.shape) == 2
    assert len(target.shape) == 2
    assert np.max(pred) <= 1
    assert np.max(target) <= 1

    pred_hard = pred.astype(int)
    target = target.astype(int)

    intersection = np.mean(pred_hard * target)
    union = np.mean(np.logical_or(pred_hard, target))
    return intersection / union


def metric_F1(pred: np.ndarray, target: np.ndarray):
    assert len(pred.shape) == 2
    assert len(target.shape) == 2
    assert np.max(pred) <= 1
    assert np.max(target) <= 1
    pred = pred.astype(int)
    target = target.astype(int)
    return f1_score(target.flatten(), pred.flatten())


def make_model_name(postfix=None):
    # Timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if postfix is None:
        postfix = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

    # Final model name
    return f"{timestamp}__{postfix}"

def combine_dicts_with_prefixes(dicts, prefixes):
    if len(dicts) != len(prefixes):
        raise ValueError("dicts and prefixes must have the same length")

    combined = {}
    for d, prefix in zip(dicts, prefixes):
        combined.update({f"{prefix}{k}": v for k, v in d.items()})
    return combined


def copy_script_folder(destination: str):
    """
    Copy all files and subfolders from the folder where this script resides
    into the given destination folder.
    """
    source = Path(__file__).parent
    destination = Path(destination)

    # Copy everything recursively, allow overwriting existing files
    shutil.copytree(source, destination, dirs_exist_ok=True)
    print("Copied training source tree into the run folder.")


def normalized_multi_scale_gradient_matching_loss(prediction: torch.Tensor, target: torch.Tensor, K=4, s: [None, torch.Tensor] = None):
    """
    Implementation of NMG loss based on MiDaS with number of scales = 4 and halfing the image resolution at each scale

    Args:
        target: (B, C, H, W)
        prediction: (B, C, H, W)
        K: number of scale levels. We use 2x2 average pooling here for better performance with a stride of 2x2 to half resolution each iteration
        s: constant scale multiplied with the prediction, shape should be (B, )
    """
    assert len(prediction.shape) == 4
    assert len(target.shape) == 4

    loss = 0
    current_scale_pred = s[:, None, None, None] * prediction if s is not None else prediction
    current_scale_target = target
    for k in range(K):
        loss_dx = torch.mean(torch.square(current_scale_target[:, :, :, 1:] - current_scale_target[:, :, :, :-1] - current_scale_pred[:, :, :, 1:] + current_scale_pred[:, :, :, :-1]))
        loss_dy = torch.mean(torch.square(current_scale_target[:, :, 1:, :] - current_scale_target[:, :, :-1, :] - current_scale_pred[:, :, 1:, :] + current_scale_pred[:, :, :-1, :]))
        loss += loss_dx + loss_dy
        if k + 1 < K:
            current_scale_pred = F.avg_pool2d(current_scale_pred, 2, 2)
            current_scale_target = F.avg_pool2d(current_scale_target, 2, 2)
    return loss


def masked_normalized_multi_scale_gradient_matching_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    K: int = 4,
    s: torch.Tensor | None = None,
):
    """
    Masked version of Normalized Multi-Scale Gradient Matching Loss (as in MiDaS).

    Args:
        prediction: (B, C, H, W)
        target:     (B, C, H, W)
        mask:       (B, H, W) boolean or {0,1} tensor; valid=1.
                    If (B,1,H,W), it will be broadcast across channels.
        K:          number of scales (2x2 avg pooling per scale).
        s:          optional scale factor (B,) applied to prediction.
    """
    assert len(prediction.shape) == 4
    assert len(target.shape) == 4
    assert len(mask.shape) == 3

    B, C, H, W = prediction.shape

    # Prepare mask
    mask = mask[:, None, :, :].float()

    current_scale_pred = s[:, None, None, None] * prediction if s is not None else prediction
    current_scale_target = target
    current_scale_mask = mask

    loss = 0.0
    for k in range(K):
        # Horizontal gradients (difference in W dimension)
        valid_dx = current_scale_mask[:, :, :, 1:] * current_scale_mask[:, :, :, :-1]
        diff_dx = current_scale_target[:, :, :, 1:] - current_scale_target[:, :, :, :-1] - current_scale_pred[:, :, :, 1:] + current_scale_pred[:, :, :, :-1]
        loss_dx = torch.mean((diff_dx ** 2 * valid_dx).sum(dim=(1, 2, 3)) / (C * valid_dx.sum(dim=(1, 2, 3)) + 1e-8))

        # Vertical gradients (difference in H dimension)
        valid_dy = current_scale_mask[:, :, 1:, :] * current_scale_mask[:, :, :-1, :]
        diff_dy = current_scale_target[:, :, 1:, :] - current_scale_target[:, :, :-1, :] - current_scale_pred[:, :, 1:, :] + current_scale_pred[:, :, :-1, :]
        loss_dy = torch.mean((diff_dy ** 2 * valid_dy).sum(dim=(1, 2, 3)) / (C * valid_dy.sum(dim=(1, 2, 3)) + 1e-8))

        loss += loss_dx + loss_dy

        # Downscale for next level
        if k + 1 < K:
            current_scale_pred = F.avg_pool2d(current_scale_pred, 2, 2)
            current_scale_target = F.avg_pool2d(current_scale_target, 2, 2)

            # mask pooling: All pixels have to be valid to be passed on
            current_scale_mask = F.avg_pool2d(current_scale_mask, 2, 2)
            current_scale_mask = (current_scale_mask > 0.9).float()

    return loss


def scale_and_shift_invariant_mean_squared_error(prediction: torch.Tensor, target: torch.Tensor, return_s_and_t=False):
    """
    https://arxiv.org/pdf/1907.01341 (just changed sums to means in the scale and shift estimate for more numerical stability)

    Input should be disparity maps (1 / depth)
    Args:
        prediction: (B, H, W)
        target: (B, H, W)
        return_s_and_t:

    Returns:

    """
    assert len(prediction.shape) == 3
    assert len(target.shape) == 3
    B, H, W = prediction.shape

    # Estimate scale and shift
    with torch.no_grad():
        left_m = torch.ones((B, 2, 2), dtype=prediction.dtype, device=prediction.device)
        left_m[:, 0, 0] = torch.sum(torch.sum(torch.square(prediction), dim=-1), dim=-1)
        left_m[:, 0, 1] = torch.sum(torch.sum(prediction, dim=-1), dim=-1)
        left_m[:, 1, 0] = left_m[:, 0, 1]
        left_m[:, 1, 1] = H * W
        left_m_inv = torch.linalg.inv(left_m)

        right_v = torch.ones((B, 2, 1), dtype=prediction.dtype, device=prediction.device)
        right_v[:, 0, 0] = torch.sum(torch.sum(prediction * target, dim=-1), dim=-1)
        right_v[:, 1, 0] = torch.sum(torch.sum(target, dim=-1), dim=-1)

        solution = left_m_inv @ right_v
        s = solution[:, 0, 0]
        t = solution[:, 1, 0]

    # Apply scale & shift
    aligned = s[:, None, None] * prediction + t[:, None, None]

    # Compute MSE
    loss = torch.mean(torch.square(aligned - target))

    if return_s_and_t:
        return loss, s, t
    return loss


def masked_scale_and_shift_invariant_mean_squared_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    return_s_and_t: bool = False
):
    """
    Masked Scale-and-Shift Invariant MSE (SSIMSE)

    Args:
        prediction: (B, H, W)
        target:     (B, H, W)
        mask:       (B, H, W) boolean or {0,1} tensor (valid = 1)
        return_s_and_t: whether to return scale and shift

    Returns:
        loss or (loss, s, t)
    """
    assert prediction.shape == target.shape == mask.shape
    B, H, W = prediction.shape

    # Ensure mask is float
    mask = mask.float()

    # Count valid pixels per batch
    valid = mask.sum(dim=(-1, -2))  # (B,)

    # Avoid division by zero
    valid = torch.clamp(valid, min=1.0)

    # Estimate scale and shift
    with torch.no_grad():
        # Compute required sums with masking
        sum_p2 = (mask * prediction * prediction).sum(dim=(-1, -2))  # Σ(p^2)
        sum_p  = (mask * prediction).sum(dim=(-1, -2))               # Σ(p)
        sum_pt = (mask * prediction * target).sum(dim=(-1, -2))      # Σ(p*t)
        sum_t  = (mask * target).sum(dim=(-1, -2))                   # Σ(t)

        # Build left matrix: [[Σ(p^2), Σ(p)],
        #                     [Σ(p),   Σ(1)]]
        left_m = torch.zeros((B, 2, 2), dtype=prediction.dtype, device=prediction.device)
        left_m[:, 0, 0] = sum_p2
        left_m[:, 0, 1] = sum_p
        left_m[:, 1, 0] = sum_p
        left_m[:, 1, 1] = valid

        # Build right vector: [Σ(p*t), Σ(t)]
        right_v = torch.zeros((B, 2, 1), dtype=prediction.dtype, device=prediction.device)
        right_v[:, 0, 0] = sum_pt
        right_v[:, 1, 0] = sum_t

        # Solve for scale (s) and shift (t)
        solution = torch.linalg.solve(left_m, right_v)
        s = solution[:, 0, 0]
        t = solution[:, 1, 0]

        # Handle cases in which s is negative
        # We essentially then have no gradient flow, but it should not cause an issue, only at the beginning of training
        s = torch.clip(s, min=0.0, max=None)

    # Apply scale and shift
    aligned = s[:, None, None] * prediction + t[:, None, None]

    # Compute masked MSE
    diff2 = (aligned - target) ** 2
    mse = (diff2 * mask).sum(dim=(-1, -2)) / valid  # per-batch
    loss = mse.mean()  # batch mean

    if return_s_and_t:
        return loss, s, t
    return loss


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """
    Computes masked MSE between prediction and target for arbitrary channels,
    using a per-pixel mask.

    Args:
        prediction:   (B, C, H, W)
        target: (B, C, H, W)
        mask:   (B, H, W) boolean or {0,1} tensor, valid=1

    Returns:
        Scalar masked MSE loss
    """
    assert prediction.shape == target.shape, "pred and target must have the same shape"

    # Convert mask to float and add channel dimension for broadcasting
    mask = mask.float().unsqueeze(1)  # (B,1,H,W), broadcast to C channels

    # Count valid pixels per batch
    valid = mask.sum(dim=(1,2,3)) * target.shape[1] # (B,)
    valid = valid.clamp_min(1.0)   # prevent division by zero

    # Compute squared error
    sq_error = (prediction - target) ** 2  # (B,C,H,W)

    # Apply mask
    masked_sq_error = sq_error * mask

    # Sum over channels and spatial dims, divide by number of valid pixels
    per_batch_loss = masked_sq_error.sum(dim=(1,2,3)) / valid  # (B,)

    # Return mean over batch
    return per_batch_loss.mean()


def masked_cross_entropy(prediction_logits: torch.Tensor, target: torch.Tensor, mask: torch.tensor, epsilon: float = 1e-6):
    """
    Args:
        prediction_logits:   (B, C, H, W)
        target:              (B, H, W) (long)
        mask:                (B, H, W) boolean or {0,1} tensor, valid=1

    Returns:
        Scalar masked CE loss
    """
    assert len(prediction_logits.shape) == 4
    assert len(target.shape) == 3
    assert len(mask.shape) == 3

    # Convert mask to float and add channel dimension for broadcasting
    mask = mask.float()

    loss_per_pos = F.cross_entropy(
        prediction_logits, target, reduction='none'
    )  # [batch, seq]

    masked_loss = (loss_per_pos * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + epsilon)

    return torch.mean(masked_loss)


def dice_loss(prediction_probabilities: torch.Tensor,
              target: torch.Tensor,
              epsilon: float = 1e-6):
    """
    Args:
        prediction_probabilities: (B, H, W) class probabilities in [0, 1]
        target:                   (B, H, W) one-hot or binary per class
        epsilon:                  small constant to avoid div-by-zero

    Returns:
        Scalar Dice loss = 1 - Dice score
    """
    assert prediction_probabilities.shape == target.shape

    # Compute per-class Dice
    intersection = (prediction_probabilities * target).sum(dim=(1, 2))
    probs_sum    = prediction_probabilities.sum(dim=(1, 2))
    target_sum   = target.sum(dim=(1, 2))

    # Compute the scores
    dice_scores = (2 * intersection + epsilon) / (probs_sum + target_sum + epsilon)

    # Final score
    dice_score = dice_scores.mean()
    return 1 - dice_score


def masked_dice_loss(prediction_probabilities: torch.Tensor,
                     target: torch.Tensor,
                     mask: torch.Tensor,
                     epsilon: float = 1e-6):
    """
    Masked Dice Loss for segmentation.

    Args:
        prediction_probabilities: (B, H, W) class probabilities in [0, 1]
        target:                   (B, H, W)  one-hot or binary per class
        mask:                     (B, H, W)  boolean or {0,1} tensor, valid=1
        epsilon:                  small constant to avoid division by zero

    Returns:
        Scalar masked Dice loss = 1 - Dice score
    """
    assert prediction_probabilities.shape == target.shape
    assert prediction_probabilities.shape == mask.shape

    # Apply mask
    prediction_probabilities = prediction_probabilities * mask
    target = target * mask

    # Compute per-class Dice
    intersection = (prediction_probabilities * target).sum(dim=(1, 2))
    probs_sum = prediction_probabilities.sum(dim=(1, 2))
    target_sum = target.sum(dim=(1, 2))

    # Compute the scores
    dice_scores = (2 * intersection + epsilon) / (probs_sum + target_sum + epsilon)

    # Final score
    dice_score = dice_scores.mean()
    return 1 - dice_score


class ResizeWithPadding:
    def __init__(self, target_size, fill=0):
        self.target_size = target_size  # (H, W)
        self.fill = fill

    def __call__(self, img):
        # Store original size (W,H)
        self.orig_size = img.size

        # Compute resize scale to fit inside target_size
        orig_w, orig_h = img.size
        target_h, target_w = self.target_size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        img = F_transforms.resize(img, (new_h, new_w), antialias=True)

        # Compute padding
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        self.padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)

        img = F_transforms.pad(img, self.padding, fill=self.fill)
        return img

    def undo(self, img):
        # Remove padding
        left, top, right, bottom = self.padding
        w, h = img.size
        img = img.crop((left, top, w - right, h - bottom))

        # Resize back to original size
        orig_w, orig_h = self.orig_size
        img = F_transforms.resize(img, (orig_h, orig_w), antialias=True)
        return img


class SquarePadToFactor:
    def __init__(self, factor=8, fill=0):
        """
        Pads each image in a batch tensor (B, C, H, W) to be square,
        with both sides divisible by `factor`. Image is centered and
        padded with `fill`.

        Args:
            factor (int): The divisor the final size must be a multiple of.
            fill (int or float): Fill value for padding.
        """
        self.factor = factor
        self.fill = fill
        self.padding = None
        self.orig_size = None

    def __call__(self, x: torch.Tensor):
        """
        x: Tensor of shape (B, C, H, W)
        """
        if x.dim() != 4:
            raise ValueError("Input must be a 4D tensor (B, C, H, W)")

        B, C, H, W = x.shape
        self.orig_size = (H, W)

        # Compute the target square size, divisible by factor
        max_side = max(H, W)
        target_side = math.ceil(max_side / self.factor) * self.factor

        # Compute required padding
        pad_h = target_side - H
        pad_w = target_side - W

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # F.pad expects (left, right, top, bottom)
        self.padding = (left, right, top, bottom)

        return F.pad(x, self.padding, value=self.fill)

    def undo(self, x):
        """
        Remove padding applied in __call__.
        """
        if self.padding is None:
            raise RuntimeError("undo() called before __call__()")

        # (B, C, H, W)
        left, right, top, bottom = self.padding
        return x[:, :, top: x.shape[2] - bottom, left: x.shape[3] - right]


def soft_cross_entropy_with_mask(logits, target_probs, mask, reduction="mean"):
    """
    logits:       (B, C, H, W) raw model outputs
    target_probs: (B, C, H, W) probability labels (sum=1 per pixel)
    mask:         (B, H, W)    1 = valid pixel, 0 = ignore pixel
    """
    # Compute log-probabilities (B, C, H, W)
    log_probs = F.log_softmax(logits, dim=1)

    # Cross entropy per pixel: -(p * log q) over classes
    loss = -(target_probs * log_probs).sum(dim=1)  # (B, H, W)

    # Apply mask
    loss = loss * mask

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        # avoid division by zero
        valid = mask.sum()
        return loss.sum() / (valid + 1e-8)
    else:
        # no reduction, return shape (B, H, W)
        return loss


class RemapFaceSyntheticsDatasetToBinaryHairSegmentation(Dataset):
    def __init__(self, dataset: FaceSyntheticsDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        valid_mask = np.zeros_like(sample['segmentation'])
        segmentation = np.zeros_like(sample['segmentation'])

        # 14: Right Brow
        segmentation[sample['segmentation'] == 5] = 1

        # 14: Left Brow
        segmentation[sample['segmentation'] == 6] = 1

        # 13: Hair
        segmentation[sample['segmentation'] == 13] = 1

        # 14: Beard
        segmentation[sample['segmentation'] == 14] = 1

        # All hairs are valid
        valid_mask[segmentation == 1] = 1

        # 15: Clothing
        valid_mask[sample['segmentation'] == 15] = 1

        # 16: Glasses
        valid_mask[sample['segmentation'] == 16] = 1

        # 17: Headwear
        valid_mask[sample['segmentation'] == 17] = 1

        # 18: Facewear
        valid_mask[sample['segmentation'] == 18] = 1


        return {
            'image': sample['image'],
            'segmentation': segmentation,
            'segmentation_mask': valid_mask
        }


class RemapHairCategoryMattesToSegmentationTargets(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        targets = self.dataset[idx]

        if 'matte' not in targets:
            return targets

        hair_categories = [
            'scalp_hair_matte', # 0
            'eyebrows_matte',   # 1
            'eyelashes_matte',  # 2
            'mustache_matte',   # 3
            'beard_matte'       # 4
        ]
        mattes = []
        for hc in hair_categories:
            if hc in targets.keys():
                mattes.append(targets[hc][:, :, None])

        if len(mattes) < len(hair_categories):
            return targets

        soft_target = np.concat(mattes, axis=2)
        hard_target = np.argmax(soft_target, axis=2)
        valid_mask = (np.max(soft_target, axis=2) > 0).astype(int)

        for hc in hair_categories:
            del targets[hc]

        targets['hair_category'] = hard_target
        targets['hair_category_mask'] = valid_mask

        return targets


class RemapHairGrowingDirectionTo2DOrientation(Dataset):
    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if 'hair_growing_direction' not in sample:
            return sample

        # Get normalized 2d projection
        v = (sample['hair_growing_direction'][:, :, :3] / 255) * 2 - 1.0
        alpha = sample['hair_growing_direction'][:, :, 3]
        xy = v[:, :, 1:].copy()
        lengths_3d = np.linalg.norm(v, axis=2)
        lengths_2d = np.linalg.norm(xy, axis=2)
        alpha_mask = alpha > 0
        alpha[alpha_mask] = 255
        mask = np.logical_and(lengths_2d > 0.01, alpha_mask)
        angles = np.arctan2(xy[:, :, 1][mask], xy[:, :, 0][mask])

        # Then convert it to double angle representation
        double_angle = 2 * angles
        xy[:, :, 0][mask] = lengths_3d[mask] * np.cos(double_angle)
        xy[:, :, 1][mask] = lengths_3d[mask] * np.sin(double_angle)

        out = np.concat([np.clip(np.round(255 * (xy * 0.5 + 0.5)), 0, 255).astype(np.uint8), sample['hair_growing_direction'][:, :, 3, None]], axis=2)
        sample['2d_orientation'] = out
        del sample['hair_growing_direction']
        return sample


class UseSynthHairAugmentations(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.augmentations = SynthHairAugmentations()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.augmentations(sample)


class ResizeMatteToImage(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.augmentations = SynthHairAugmentations()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if 'matte' not in sample or 'image' not in sample:
            return sample

        sample['matte'] = cv2.resize(sample['matte'], (int(sample['image'].shape[1]), int(sample['image'].shape[0])), interpolation=cv2.INTER_LINEAR)
        return sample


class RemapSegmentationToMatteTarget(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if 'segmentation' not in sample:
            return sample

        sample['matte'] = (sample['segmentation'] * 255).astype(np.uint8)
        del sample['segmentation']
        return sample
