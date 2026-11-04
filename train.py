#!/usr/bin/env python3
"""Train using configs/paths.yaml (fixed location) and a training YAML config (any path or preset name)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
PATHS_CONFIG = REPO_ROOT / "configs" / "paths.yaml"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hair matting training launcher")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="PATH_OR_NAME",
        help="Training YAML: absolute path, or path relative to cwd / repo root, or preset name "
        "(e.g. Matte512 → configs/train/Matte512.yaml)",
    )
    parser.add_argument("--device", type=str, default=None, help="Override training device from the preset YAML")
    parser.add_argument("--out-folder", type=str, default=None, help="Override run output directory from the preset YAML")
    args = parser.parse_args()

    if not PATHS_CONFIG.is_file():
        raise FileNotFoundError(
            f"Missing dataset paths file: {PATHS_CONFIG}. Add this file next to the repo (see comments inside for layout)."
        )

    from datasets import apply_paths_yaml

    apply_paths_yaml(PATHS_CONFIG)

    from trainer import (
        load_training_config_yaml,
        merge_overrides,
        resolve_train_config_path,
        run_train,
    )

    cfg_path = resolve_train_config_path(REPO_ROOT, args.config)
    config = load_training_config_yaml(cfg_path)
    config = merge_overrides(config, device=args.device, out_folder=args.out_folder)
    run_train(config)


if __name__ == "__main__":
    main()
