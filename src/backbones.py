import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

image_normalize_dino_v2 = transforms.Compose([
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class GlobalFeatureExtractor(nn.Module):
    def __init__(self, backbone='dinov2_vits14', extra_patch_levels=3):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', backbone)
        self.embed_dim = self.backbone.embed_dim
        self.extra_patch_levels = extra_patch_levels
        self.dino_input_res = 518

    def get_dino_features(self, img):
        b, c, h, w = img.shape
        patch_size = self.backbone.patch_size
        dino_features = self.backbone.forward_features(img)['x_norm_patchtokens']
        dino_features = dino_features.reshape(b, h // patch_size, w // patch_size, -1)
        dino_features = dino_features.permute(0, 3, 1, 2)  # B, C, H, W
        return dino_features

    def get_patches(self, img, num_patches_y=1, num_patches_x=1):
        if num_patches_y == 1 and num_patches_x == 1:
            return F.interpolate(img, (self.dino_input_res, self.dino_input_res), mode='area')

        b, c, h, w = img.shape
        patch_h = h // num_patches_y
        patch_w = w // num_patches_x

        patches = []
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                patch = img[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]
                patches.append(patch)

        # (num_patches, B, C, ph, pw) -> (B, num_patches, ...) so each image's patches stay together
        stacked = torch.stack(patches, dim=0).permute(1, 0, 2, 3, 4)
        stacked = stacked.reshape(b * len(patches), c, patch_h, patch_w)
        return F.interpolate(stacked, (self.dino_input_res, self.dino_input_res), mode='area')

    def reassemble(self, x, batch_size, num_patches_y=1, num_patches_x=1, resize=None):
        if num_patches_y != 1 or num_patches_x != 1:
            _, c, h, w = x.shape
            x = x.reshape(batch_size, num_patches_y, num_patches_x, c, h, w)
            x = x.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, c, h * num_patches_y, w * num_patches_x)
        if resize:
            x = F.interpolate(x, resize, mode='bilinear')
        return x

    def _multiscale_features(self, img):
        batch_size = img.shape[0]
        levels = [(1, 1)]
        if self.extra_patch_levels >= 1:
            levels.append((2, 2))
        if self.extra_patch_levels >= 2:
            levels.append((4, 4))
        packed = torch.concat([self.get_patches(img, ny, nx) for ny, nx in levels], dim=0)
        features = self.get_dino_features(packed)
        target_size = int(round(self.dino_input_res / self.backbone.patch_size)) * 2 ** self.extra_patch_levels
        offset = 0
        scales = []
        for index, (ny, nx) in enumerate(levels):
            count = batch_size * ny * nx
            resize = None if index == len(levels) - 1 else (target_size, target_size)
            scales.append(self.reassemble(features[offset:offset + count], batch_size, ny, nx, resize))
            offset += count
        if len(scales) == 2:
            return 0.5 * (scales[0] + scales[1])
        return (1 / 3) * (scales[0] + scales[1] + scales[2])

    def forward(self, img):
        img = image_normalize_dino_v2(img)
        if self.extra_patch_levels == 0:
            x = self.get_dino_features(self.get_patches(img))
        elif self.extra_patch_levels in (1, 2):
            x = self._multiscale_features(img)
        else:
            raise ValueError("extra_patch_levels greater than 2 are not supported")

        return x  # (B, self.embed_dim, H, W)


class GlobalIntermediateFeatureExtractor(nn.Module):
    def __init__(self, backbone='dinov2_vits14'):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', backbone)
        self.embed_dim = self.backbone.embed_dim
        self.intermediate_features = [None, None, None, None]
        self.input_resolution = (518, 518)

        intermediate_layer_idx = {
            'dinov2_vits14': [2, 5, 8, 11],
            'dinov2_vitb14': [2, 5, 8, 11],
            'dinov2_vitl14': [4, 11, 17, 23],
            'dinov2_vitg14': [9, 19, 29, 39]
        }

        self.setup_hooks(intermediate_layer_idx[backbone])
        # self.embed_dim = self.backbone.embed_dim

    def setup_hooks(self, intermediate_layer_idx):
        for i, block_idx in enumerate(intermediate_layer_idx):
            self.backbone.blocks[block_idx].register_forward_hook(self.get_intermediate_feature_hook(i))

    def get_intermediate_feature_hook(self, name):
        def hook(module, input, output):
            patch_size = self.backbone.patch_size
            readout = output[:, 1:, :]
            readout = readout.reshape(readout.shape[0], self.input_resolution[1] // patch_size, self.input_resolution[0] // patch_size, -1)
            readout = readout.permute(0, 3, 1, 2)  # B, C, H, W
            self.intermediate_features[name] = readout
        return hook

    def forward(self, img):
        img = image_normalize_dino_v2(img)
        img = F.interpolate(img, self.input_resolution, mode='area')
        self.intermediate_features = [None, None, None, None]
        self.backbone.forward_features(img)
        return self.intermediate_features  # list of (B, self.embed_dims[idx in list], H, W)


class ResBlock(nn.Module):
    def __init__(self, ch, mid_ch):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch, mid_ch, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(mid_ch, ch, 3, padding='same')
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.main(x)


class DPImageEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_res_blocks=1):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False))
        for _ in range(num_res_blocks):
            self.enc.append(ResBlock(out_ch, out_ch // 2))
        self.pooling = nn.AvgPool2d(2)

    def forward(self, x):
        enc_features = self.enc(x)
        next_res_features = self.pooling(enc_features)
        return enc_features, next_res_features


class DPBackboneEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_res_blocks=1):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False))
        for _ in range(num_res_blocks):
            self.enc.append(ResBlock(out_ch, out_ch // 2))

    def forward(self, x):
        return self.enc(x)


class DPFusionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_res_blocks=1):
        super().__init__()
        self.res_blocks = nn.Sequential()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResBlock(in_ch, in_ch // 2))
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x, image_features, backbone_features):
        B, C, H, W = image_features.shape

        backbone_features = F.interpolate(backbone_features, (H, W), mode='bilinear')

        if x is not None:
            x = F.interpolate(x, (H, W), mode='bilinear')
            x = x + image_features + backbone_features
        else:
            x = image_features + backbone_features

        x = self.res_blocks(x)
        x = self.proj(x)
        return x


class DensePredictionHead(nn.Module):
    def __init__(self, backbone_dim, output_head_dim, unet_levels=3, residual_unit_base_width=8, residual_unit_depth=1):
        super().__init__()
        self.image_enc_blocks = nn.Sequential()
        self.backbone_enc_blocks = nn.Sequential()
        self.fusion_blocks = nn.Sequential()
        self.prediction_head = nn.Conv2d(residual_unit_base_width, output_head_dim, 1)
        for i in range(unet_levels):
            self.image_enc_blocks.append(
                DPImageEncoderBlock(
                    in_ch=3 if i == 0 else residual_unit_base_width * 2 ** (i - 1),
                    out_ch=residual_unit_base_width * 2 ** i,
                    num_res_blocks=residual_unit_depth
                )
            )

            self.backbone_enc_blocks.append(
                DPBackboneEncoderBlock(
                    in_ch=backbone_dim,
                    out_ch=residual_unit_base_width * 2 ** i,
                    num_res_blocks=residual_unit_depth
                )
            )

            self.fusion_blocks.append(
                DPFusionBlock(
                    in_ch=residual_unit_base_width * 2 ** i,
                    out_ch=residual_unit_base_width * 2 ** max(i - 1, 0),
                )
            )

    def forward(self, img: torch.Tensor, backbone: [list, torch.Tensor]):

        # First, we encode the image features
        x = image_normalize_dino_v2(img)
        image_features = []
        for block in self.image_enc_blocks:
            image_f, x = block(x)
            image_features.append(image_f)

        # Then call fusion and backbone in reverse order
        x = None
        for stage_idx, (fusion_block, backbone_block, image_f) in enumerate(reversed(list(zip(self.fusion_blocks, self.backbone_enc_blocks, image_features)))):
            backbone_f = backbone_block(backbone[stage_idx] if isinstance(backbone, list) else backbone)
            x = fusion_block(x, image_f, backbone_f)

        # Then a final prediction head
        x = self.prediction_head(x)
        return x
