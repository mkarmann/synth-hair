import os

import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from backbones import DensePredictionHead, GlobalFeatureExtractor, GlobalIntermediateFeatureExtractor
from utils import SquarePadToFactor

HF_REPO_ID = "StableMarkus/synth-hair-h2m2"
_HUB_DOWNLOAD_KEYS = ("revision", "cache_dir", "force_download", "token", "local_files_only")


def _build_backbone_and_head(output_head_dim: int, backbone: str, resnet_width, resnet_stages, resnet_depth, backbone_extra_patch_levels, build_dpt=False):
    if build_dpt:
        backbone_model = GlobalIntermediateFeatureExtractor(
            backbone=backbone
        )
        dense_prediction_head = DensePredictionHead(
            backbone_dim=backbone_model.embed_dim,
            output_head_dim=output_head_dim,
            residual_unit_base_width=resnet_width,
            unet_levels=len(backbone_model.intermediate_features),
            residual_unit_depth=resnet_depth
        )
    else:
        # We take the feature output and feed it directly to the model
        backbone_model = GlobalFeatureExtractor(
            backbone=backbone,
            extra_patch_levels=backbone_extra_patch_levels
        )

        dense_prediction_head = DensePredictionHead(
            backbone_dim=backbone_model.embed_dim,
            output_head_dim=output_head_dim,
            residual_unit_base_width=resnet_width,
            unet_levels=resnet_stages,
            residual_unit_depth=resnet_depth
        )

    return backbone_model, dense_prediction_head


class DensePredictionModel(nn.Module, PyTorchModelHubMixin, repo_url="https://github.com/mkarmann/synth-hair", license="cc-by-nc-4.0"):
    def __init__(self, output_head_dim: int, backbone='dinov2_vits14', resnet_width=16, resnet_stages=3, resnet_depth=8, backbone_extra_patch_levels=0, build_dpt=False):
        super().__init__()
        self.resnet_stages = resnet_stages
        self.backbone, self.dense_prediction_head = _build_backbone_and_head(
            output_head_dim=output_head_dim,
            backbone=backbone,
            resnet_width=resnet_width,
            resnet_depth=resnet_depth,
            resnet_stages=resnet_stages,
            backbone_extra_patch_levels=backbone_extra_patch_levels,
            build_dpt=build_dpt
        )

    def forward(self, img, no_grad_backbone=True):
        padding = SquarePadToFactor(2**(self.resnet_stages - 1))
        img = padding(img)
        if no_grad_backbone:
            with torch.no_grad():
                x = self.backbone(img)
        else:
            x = self.backbone(img)
        x = self.dense_prediction_head(img, x)
        out = padding.undo(x)
        return out

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, subfolder=None, strict=True, **kwargs):
        path = str(pretrained_model_name_or_path)
        if subfolder:
            if os.path.isdir(path):
                path = os.path.join(path, subfolder)
            else:
                download_kwargs = {key: kwargs[key] for key in _HUB_DOWNLOAD_KEYS if key in kwargs}
                config_file = hf_hub_download(repo_id=path, filename="config.json", subfolder=subfolder, **download_kwargs)
                hf_hub_download(repo_id=path, filename="model.safetensors", subfolder=subfolder, **download_kwargs)
                path = os.path.dirname(config_file)
        return super().from_pretrained(path, strict=strict, **kwargs)

    @classmethod
    def _from_pretrained(cls, *, model_id, revision, cache_dir, force_download, local_files_only, token, map_location="cpu", strict=False, **model_kwargs):
        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            model_file = os.path.join(model_id, "model.safetensors")
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename="model.safetensors",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                local_files_only=local_files_only,
            )
        return cls._load_as_safetensor(model, model_file, map_location, strict)


class AllLabelsModel(DensePredictionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(1 + 5 + 3 + 1, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        out = dict()
        out['matte'] = torch.sigmoid(x[:, 0])  # [0, 1] (B, H, W)
        out['hair_category'] = x[:, 1:6]  # Logits (B, 5, H, W)
        out['hair_growing_direction'] = torch.sigmoid(x[:, 6:9])  # [0, 1] (B, 3, H, W)
        out['depth'] = x[:, 9]  # (B, H, W)
        return out


class All2DLabelsModel(DensePredictionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(1 + 5 + 2 + 1, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        out = dict()
        out['matte'] = torch.sigmoid(x[:, 0])  # [0, 1] (B, H, W)
        out['hair_category'] = x[:, 1:6]  # Logits (B, 5, H, W)
        out['2d_orientation'] = torch.sigmoid(x[:, 6:8])  # [0, 1] (B, "3", H, W)
        out['depth'] = x[:, 8]  # (B, H, W)
        return out


class HairMatteModel(DensePredictionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        out = dict()
        out['matte'] = torch.sigmoid(x[:, 0])  # [0, 1] (B, H, W)
        return out


class HairCategoryModel(DensePredictionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(5, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        out = dict()
        out['hair_category'] = x # Logits (B, 5, H, W)
        return out


class HairGrowthDirectionModel(DensePredictionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        out = dict()
        out['hair_growing_direction'] = torch.sigmoid(x)  # [0, 1] (B, 3, H, W)
        return out


class DepthModel(DensePredictionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        out = dict()
        out['depth'] = x[:, 0]  # [0, 1] (B, 3, H, W)
        return out


class HairAttributesModel(DensePredictionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(5 + 3 + 1, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        out = dict()
        out['hair_category'] = x[:, 0:5]  # Logits (B, 5, H, W)
        out['hair_growing_direction'] = torch.sigmoid(x[:, 5:8])  # [0, 1] (B, 3, H, W)
        out['depth'] = x[:, 8]  # (B, H, W)
        return out


class Hair2DAttributesModel(DensePredictionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(5 + 2 + 1, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        out = dict()
        out['hair_category'] = x[:, 0:5]  # Logits (B, 5, H, W)
        out['2d_orientation'] = torch.sigmoid(x[:, 5:7])  # [0, 1] (B, 2, H, W)
        out['depth'] = x[:, 7]  # (B, H, W)
        return out


def load_hf_model(name: str):
    key = name.removeprefix("hf:")
    if key == "matte":
        return HairMatteModel.from_pretrained(HF_REPO_ID, subfolder="matte")
    if key == "attributes":
        return Hair2DAttributesModel.from_pretrained(HF_REPO_ID, subfolder="attributes")
    raise ValueError(f'Unknown released checkpoint "{name}". Use hf:matte or hf:attributes.')
