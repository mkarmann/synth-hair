import argparse
import json
import os
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import tifffile
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import ToPILImage

from model import load_hf_model
from trainer import load_model_from_args


def save_predictions(out_folder: str, fname: str, prediction: dict):
    fname_wo_dtype = fname.split('.')[0]
    for k, v in prediction.items():
        if k == 'matte':
            matte = ToPILImage()(np.clip(v[0].float().cpu().numpy(), 0, 1))
            os.makedirs(os.path.join(out_folder, 'matte'), exist_ok=True)
            matte.save(os.path.join(out_folder, 'matte', fname_wo_dtype + '.png'))
        elif k == 'hair_category':
            hair_category_arr = torch.softmax(v[0], dim=0).float().cpu().numpy().transpose((1, 2, 0))
            os.makedirs(os.path.join(out_folder, 'hair_category'), exist_ok=True)
            np.save(os.path.join(out_folder, 'hair_category', fname_wo_dtype + '.npy'), hair_category_arr)
        elif k == 'hair_growing_direction':
            hair_growing_direction = ToPILImage()(np.clip(v[0].float().cpu().numpy().transpose((1, 2, 0)), 0, 1))
            os.makedirs(os.path.join(out_folder, 'hair_growing_direction'), exist_ok=True)
            hair_growing_direction.save(os.path.join(out_folder, 'hair_growing_direction', fname_wo_dtype + '.png'))
        elif k == '2d_orientation':
            xy = np.clip(v[0].float().cpu().numpy().transpose((1, 2, 0)), 0, 1)
            orientation = ToPILImage()(np.concat([xy, xy[:, :, 0, None] * 0], axis=2))
            os.makedirs(os.path.join(out_folder, '2d_orientation'), exist_ok=True)
            orientation.save(os.path.join(out_folder, '2d_orientation', fname_wo_dtype + '.png'))
        elif k == 'depth':
            depth = v[0].float().cpu().numpy()
            depth = depth - np.min(depth)
            depth = depth / (np.max(depth) + 0.0000001)
            os.makedirs(os.path.join(out_folder, 'depth'), exist_ok=True)
            tifffile.imwrite(os.path.join(out_folder, 'depth', fname_wo_dtype + '.tiff'), depth, compression="lzw")
        else:
            print(f'Warning: Got unknown prediction name "{k}"')



def main():
    parser = argparse.ArgumentParser(description="Predicting images")
    parser.add_argument("--input-path", required=True, type=str, help="Input folder path")
    parser.add_argument("--output-path", default="runs/inference_output", type=str, help="Output folder path")
    parser.add_argument("--ckpt-path", default="hf:matte", type=str, help='Checkpoint. "hf:matte" or "hf:attributes" loads the released weights. A local .pth also reads args.json beside that file.')
    parser.add_argument("--device", default="cuda:0", type=str, help="PyTorch Device")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    if args.ckpt_path.startswith("hf:"):
        model = load_hf_model(args.ckpt_path)
    else:
        with open(os.path.join(os.path.dirname(args.ckpt_path), "args.json"), "r") as f:
            data = json.load(f)
            model_args = argparse.Namespace(**data)

        model = load_model_from_args(model_args)
        model.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device('cpu')))
    model.to(args.device)
    model.to(torch.bfloat16)
    model.eval()

    with torch.no_grad():
        for fname in tqdm(list(sorted(os.listdir(args.input_path))), 'Predicting'):
            im_filepath = os.path.join(args.input_path, fname)
            im = Image.open(im_filepath).convert("RGB")
            im_tensor = transforms.ToTensor()(im).float().to(args.device).to(torch.bfloat16)
            prediction = model(im_tensor[None])
            save_predictions(args.output_path, fname, prediction)


if __name__ == '__main__':
    main()
