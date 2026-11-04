import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import yaml
from model import HairMatteModel, DepthModel, HairGrowthDirectionModel, HairCategoryModel, AllLabelsModel, HairAttributesModel, Hair2DAttributesModel, All2DLabelsModel
from utils import make_model_name, set_all_seeds, copy_script_folder, normalized_multi_scale_gradient_matching_loss, masked_scale_and_shift_invariant_mean_squared_error, masked_normalized_multi_scale_gradient_matching_loss, masked_mse, masked_cross_entropy, RemapHairCategoryMattesToSegmentationTargets, RemapHairGrowingDirectionTo2DOrientation, UseSynthHairAugmentations, RemapSegmentationToMatteTarget, ResizeMatteToImage
from datasets import SynthHairDataset, Figaro1kDataset, HiSa, CelebAMaskHQDataset, synthetic_hair_root_for_resolution
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt


TrainDataset = Literal["SynthHair", "HiSa", "Figaro1k", "CelebAMaskHQ", "CelebAMaskHQScalpHair"]
ModelType = Literal["matte", "hair_category", "hair_growing_direction", "depth", "all_attributes", "all_attributes_2d", "all", "all_2d"]


@dataclass
class TrainingConfig:
    name: str
    out_folder: str
    device: str = "cuda:0"
    train_dataset: TrainDataset = "SynthHair"
    training_resolution: int = 512
    epochs: int = 15
    batch_size: int = 1
    backbone_learning_rate: float = 0.0
    head_learning_rate: float = 0.0001
    reduce_lr_after_each: int = 10
    reduce_lr_by_a_factor_of: float = 0.1
    continue_from_ckpt: str | None = None
    seed: int = 2025
    use_training_augmentation: bool = True
    build_dpt: bool = False
    backbone: str = "dinov2_vits14"
    model_type: ModelType = "matte"
    resnet_width: int = 8
    resnet_depth: int = 8
    resnet_stages: int = 4
    backbone_extra_patch_levels: int = 2

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(**asdict(self))


def training_config_from_dict(data: dict) -> TrainingConfig:
    d = dict(data)
    if "train_dataset" not in d:
        d["train_dataset"] = "SynthHair"
    field_names = {f.name for f in fields(TrainingConfig)}
    unknown = set(d) - field_names
    for k in unknown:
        del d[k]
    return TrainingConfig(**d)


def load_training_config_yaml(path: Path) -> TrainingConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Empty YAML: {path}")
    if isinstance(raw, dict) and "train" in raw and isinstance(raw["train"], dict):
        payload = raw["train"]
    else:
        payload = raw
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return training_config_from_dict(payload)


def merge_overrides(
    config: TrainingConfig,
    *,
    device: str | None = None,
    out_folder: str | None = None,
) -> TrainingConfig:
    d = asdict(config)
    if device is not None:
        d["device"] = device
    if out_folder is not None:
        d["out_folder"] = out_folder
    return TrainingConfig(**d)


def resolve_train_config_path(repo_root: Path, config_arg: str) -> Path:
    """Resolve a train config: explicit file path (anywhere), then repo presets."""
    raw = Path(config_arg).expanduser()
    # 1) As given: absolute, or relative to current working directory
    if raw.is_file():
        return raw.resolve()
    # 2) Relative to repository root (handy when cwd is not the repo)
    under_repo = (repo_root / raw).resolve()
    if under_repo.is_file():
        return under_repo
    # 3) Preset name / filename under configs/train/
    cand = repo_root / "configs" / "train" / config_arg
    if cand.is_file():
        return cand.resolve()
    if not str(config_arg).endswith(".yaml") and not str(config_arg).endswith(".yml"):
        cand = repo_root / "configs" / "train" / f"{config_arg}.yaml"
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"Train config not found: {config_arg!r}. Tried cwd-relative path, "
        f"repo-relative ({repo_root / raw}), and configs/train/ presets."
    )


def plot_prediction_and_target_tensor(img, pred, target, title=''):
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True)
    p = pred.detach().cpu().numpy()[0]
    t = target.detach().cpu().numpy()[0]
    print(p.shape, t.shape)
    if len(p.shape) == 3:
        if p.shape[0] == 1 or p.shape[0] == 3 or p.shape[0] == 4:
            ax1.imshow(p.transpose(1, 2, 0))
        elif p.shape[0] == 2:
            ax1.imshow(p[:, :, 0].transpose(1, 2, 0))
        else:
            ax1.imshow(np.argmax(p, axis=2).astype(float))
    else:
        ax1.imshow(p)
    if len(t.shape) == 3:
        if t.shape[0] == 1 or t.shape[0] == 3 or t.shape[0] == 4:
            ax2.imshow(t.transpose(1, 2, 0))
        elif t.shape[0] == 2:
            ax2.imshow(t[:, :, 0].transpose(1, 2, 0))
        else:
            ax2.imshow(np.argmax(t, axis=2).astype(float))
    else:
        ax2.imshow(t)
    ax3.imshow(img.detach().cpu().numpy()[0].transpose((1, 2, 0)))
    ax1.set_title(title)
    ax2.set_title('GT')
    plt.show()


def estimate_losses(targets: dict, predictions: dict, verbose=0):
    losses = {
        'total': 0.0
    }

    num_of_tasks = 0

    for k, pred in predictions.items():
        if k not in targets.keys():
            continue

        # Get the target
        target = targets[k]

        if verbose > 0:
            plot_prediction_and_target_tensor(targets['image'], pred, target, title=k)

        # Matting
        if k == 'matte':
            mse_loss = F.mse_loss(pred, target)
            gradient_loss = normalized_multi_scale_gradient_matching_loss(pred[:, None], target[:, None])
            losses['matte_mse_loss'] = mse_loss
            losses['matte_gradient_loss'] = gradient_loss
            losses['total'] = losses['total'] + 0.5 * mse_loss + 0.5 * gradient_loss
            num_of_tasks += 1

        elif k == 'hair_category':
            if 'hair_category_mask' in targets:
                mask = targets['hair_category_mask']
                ce_loss = masked_cross_entropy(pred, target, mask)
                # d_loss = 0
                # for i in range(5):
                #     d_loss = d_loss + masked_dice_loss(pred_probabilities[:, i], (target == i).long(), mask) / 5
            else:
                ce_loss = F.cross_entropy(pred, target, reduction='mean')
                # d_loss = 0
                # for i in range(5):
                #     d_loss = d_loss + dice_loss(pred_probabilities[:, i], (target == i).long()) / 5
            losses['hair_category_cross_entropy_loss'] = ce_loss      # 0.01 to get values in same range as other losses
            # losses['hair_category_dice_loss'] = d_loss
            losses['total'] = losses['total'] + 0.1 * ce_loss
            num_of_tasks += 1

        # Hair growing direction
        elif k == 'hair_growing_direction':
            mask = target[:, 3, :, :] > 0
            # _, axes = plt.subplots(1, 2, sharex=True, sharey=True)
            # axes[0].imshow(mask[0].cpu().numpy())
            # axes[1].imshow(target[0, :3].permute(1, 2, 0).cpu().numpy())
            # plt.show()
            mse_loss = masked_mse(pred, target[:, :3, :, :], mask)
            gradient_loss = masked_normalized_multi_scale_gradient_matching_loss(pred, target[:, :3, :, :], mask)
            losses['hair_growing_direction_mse_loss'] = mse_loss
            losses['hair_growing_direction_gradient_loss'] = gradient_loss
            losses['total'] = losses['total'] + (0.5 * mse_loss + 0.5 * gradient_loss)
            num_of_tasks += 1

        # Hair 2d orientation
        elif k == '2d_orientation':
            mask = target[:, 2, :, :] > 0
            # _, axes = plt.subplots(1, 2, sharex=True, sharey=True)
            # axes[0].imshow(mask[0].cpu().numpy())
            # axes[1].imshow(target[0, :3].permute(1, 2, 0).cpu().numpy())
            # plt.show()
            mse_loss = masked_mse(pred, target[:, :2, :, :], mask)
            gradient_loss = masked_normalized_multi_scale_gradient_matching_loss(pred, target[:, :2, :, :], mask)
            losses['2d_orientation_mse_loss'] = mse_loss
            losses['2d_orientation_gradient_loss'] = gradient_loss
            losses['total'] = losses['total'] + (0.5 * mse_loss + 0.5 * gradient_loss) * 0.1
            num_of_tasks += 1

        # Depth
        elif k == 'depth':
            mask = torch.logical_and(target != 0, targets['matte'] > 0)
            target_inverse_depth = target.clone()
            target_inverse_depth[mask] = 1 / target_inverse_depth[mask]

            # Normalize inverse depth (disparity) to range 0 to 1
            target_min = target_inverse_depth[mask].min()
            target_max = target_inverse_depth[mask].max()
            target_inverse_depth[mask] = (target_inverse_depth[mask] - target_min) / (target_max - target_min)

            # Calculate losses
            ssimse_loss, s, t = masked_scale_and_shift_invariant_mean_squared_error(pred, target_inverse_depth, mask, return_s_and_t=True)
            gradient_loss = masked_normalized_multi_scale_gradient_matching_loss(pred[:, None], target_inverse_depth[:, None], mask, s=s)
            losses['depth_ssimse_loss'] = ssimse_loss
            losses['depth_grad_loss'] = gradient_loss
            losses['total'] = losses['total'] + 0.5 * ssimse_loss + 0.5 * gradient_loss
            num_of_tasks += 1

        else:
            raise Exception('Unknown prediction type' + k)

    if num_of_tasks > 1:
        losses['total'] = losses['total'] / num_of_tasks

    return losses


def batch_to_gpu_tensor(batch, device):
    batch_gpu = dict()
    for k, v in batch.items():
        if k in ['segmentation', 'segmentation_mask', 'hair_category', 'hair_category_mask']:
            batch_gpu[k] = v.long().to(device)
            # plt.imshow(batch_gpu[k].cpu().numpy()[0])
            # plt.show()
        elif k in ['matte', 'scalp_hair_matte', 'eyebrows_matte', 'eyelashes_matte', 'mustache_matte', 'beard_matte']:
            batch_gpu[k] = (v.float() / 255).to(device)
        elif k in ['image', 'hair_growing_direction', '2d_orientation']:
            batch_gpu[k] = (v.float() / 255).to(device).permute(0, 3, 1, 2)
        elif k in ['depth']:
            batch_gpu[k] = v.float().to(device)
    return batch_gpu


def load_model_from_args(args):
    kwargs = {k: v for k, v in vars(args).items() if k in ["backbone_extra_patch_levels", "resnet_width", "resnet_stages", "resnet_depth", "build_dpt", "backbone"]}
    if args.model_type == 'matte':
        model = HairMatteModel(**kwargs)
    elif args.model_type == 'depth':
        model = DepthModel(**kwargs)
    elif args.model_type == 'hair_growing_direction':
        model = HairGrowthDirectionModel(**kwargs)
    elif args.model_type == 'hair_category':
        model = HairCategoryModel(**kwargs)
    elif args.model_type == 'all':
        model = AllLabelsModel(**kwargs)
    elif args.model_type == 'all_2d':
        model = All2DLabelsModel(**kwargs)
    elif args.model_type == 'all_attributes':
        model = HairAttributesModel(**kwargs)
    elif args.model_type == 'all_attributes_2d':
        model = Hair2DAttributesModel(**kwargs)
    else:
        exit('Unsupported model type')

    return model


def run_train(config: TrainingConfig) -> None:
    args = config.to_namespace()
    print()
    print("--------------")
    print("Configuration:")
    print("--------------")
    for key, value in vars(args).items():
        print(f"{key} = {value}")
    if args.backbone_learning_rate == 0.0:
        set_all_seeds(args.seed)

    # -------------
    # Output Folder
    # -------------
    run_name = make_model_name(postfix=args.name)
    run_folder = os.path.abspath(os.path.join(args.out_folder, run_name))
    print("Output folder:", run_folder)
    os.makedirs(run_folder)
    copy_script_folder(run_folder)
    writer = SummaryWriter(log_dir=run_folder)

    # -----
    # Model
    # -----
    model = load_model_from_args(args)
    if args.continue_from_ckpt is not None:
        model.load_state_dict(torch.load(args.continue_from_ckpt))
    model = model.to(args.device)

    # --------
    # Datasets
    # --------
    if args.train_dataset == "SynthHair":
        train_dataset = RemapHairCategoryMattesToSegmentationTargets(
            SynthHairDataset(
                data_path=synthetic_hair_root_for_resolution(args.training_resolution),
                split="train",
                use_train_augmentation=args.use_training_augmentation,
                load_only_hair_matte=args.model_type == "matte",
            )
        )
    elif args.train_dataset == "HiSa":
        train_dataset = UseSynthHairAugmentations(
            HiSa(
                split="train"
            )
        )
    elif args.train_dataset == "Figaro1k":
        train_dataset = UseSynthHairAugmentations(
            RemapSegmentationToMatteTarget(Figaro1kDataset(
                split="train"
            ))
        )
    elif args.train_dataset == "CelebAMaskHQ":
        random.seed(2025)
        ids = list(np.arange(len(CelebAMaskHQDataset())))
        random.shuffle(ids)
        train_dataset = UseSynthHairAugmentations(
            Subset(
                ResizeMatteToImage(RemapSegmentationToMatteTarget(CelebAMaskHQDataset())),
                ids[10000:]
            )
        )
    elif args.train_dataset == "CelebAMaskHQScalpHair":
        random.seed(2025)
        ids = list(np.arange(len(CelebAMaskHQDataset(load_just_hair=False, load_just_scalp_hair=True))))
        random.shuffle(ids)
        train_dataset = UseSynthHairAugmentations(
            Subset(
                ResizeMatteToImage(RemapSegmentationToMatteTarget(CelebAMaskHQDataset(load_just_hair=False, load_just_scalp_hair=True))),
                ids[10000:]
            )
        )
    else:
        exit("Unknown training dataset " + args.train_dataset)

    if args.model_type == "all_attributes_2d":
        train_dataset = RemapHairGrowingDirectionTo2DOrientation(train_dataset)

    run_config = dict(vars(args))
    with open(os.path.join(run_folder, "args.json"), "w") as f:
        json.dump(run_config, f, indent=4)
    with open(os.path.join(run_folder, "args.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    # -----------
    # Dataloaders
    # -----------
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    head_optimizer = torch.optim.Adam(model.dense_prediction_head.parameters(), lr=args.head_learning_rate)
    backbone_optimizer = torch.optim.Adam(model.backbone.parameters(), lr=args.backbone_learning_rate) if args.backbone_learning_rate > 0 else None
    head_lr_scheduler = torch.optim.lr_scheduler.StepLR(head_optimizer, step_size=args.reduce_lr_after_each, gamma=args.reduce_lr_by_a_factor_of)
    backbone_lr_scheduler = torch.optim.lr_scheduler.StepLR(backbone_optimizer, step_size=args.reduce_lr_after_each, gamma=args.reduce_lr_by_a_factor_of) if backbone_optimizer is not None else None

    # -------------
    # Training loop
    # -------------
    global_step = 0
    for epoch in range(args.epochs):

        # -----
        # Train
        # -----
        model.train()
        if backbone_optimizer is None:
            model.backbone.eval()
        for batch_cpu in (pbar := tqdm(train_dataloader, f"Epoch {epoch+1} (cuda: {args.device})")):
            global_step += 1

            batch_gpu = batch_to_gpu_tensor(batch_cpu, args.device)
            predictions = model(batch_gpu["image"], no_grad_backbone=backbone_optimizer is None)
            losses = estimate_losses(batch_gpu, predictions)
            total_loss = losses["total"]
            total_loss.backward()
            for k, v in losses.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

            # Write to console
            pbar.set_postfix_str(f"total: {total_loss:0.5f}")

            # Log extra information
            writer.add_scalar("Model/Epoch", epoch, global_step)
            writer.add_scalar("Model/HeadLR", head_optimizer.param_groups[0]["lr"], global_step)
            if backbone_optimizer is not None:
                writer.add_scalar("Model/BackboneLR", backbone_optimizer.param_groups[0]["lr"], global_step)

            # Step
            head_optimizer.step()
            head_optimizer.zero_grad(set_to_none=True)
            if backbone_optimizer is not None:
                backbone_optimizer.step()
                backbone_optimizer.zero_grad(set_to_none=True)

        # Update LR schedule
        head_lr_scheduler.step()
        if backbone_lr_scheduler is not None:
            backbone_lr_scheduler.step()

        # Save the model
        torch.save(model.state_dict(), os.path.join(run_folder, "model_latest.pth"))
        torch.save(model.state_dict(), os.path.join(run_folder, f"model_epoch_{epoch + 1}.pth"))


if __name__ == "__main__":
    raise SystemExit(
        "Use the repo-root train.py entrypoint (paths + YAML), e.g.:\n"
        "  python train.py --config Matte512"
    )
