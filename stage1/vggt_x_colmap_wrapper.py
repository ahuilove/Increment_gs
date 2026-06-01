#!/usr/bin/env python3
"""
Run VGGT-X COLMAP export from the main project without modifying the submodule.

This wrapper reuses submodules/vggt-x/demo_colmap.py, but replaces its model
loading function so the VGGT checkpoint is read from a local file under stage1.

Example:
    python stage1/vggt_x_colmap_wrapper.py \
        --scene_dir stage1/chunk/chunk_0000 \
        --checkpoint stage1/checkpoint/vggt_1b_model.pt \
        --use_ga \
        --shared_camera \
        --chunk_size 128 \
        --max_points_for_colmap 500000
"""

from __future__ import annotations


import argparse
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
VGGT_X_ROOT = REPO_ROOT / "submodules" / "vggt-x"
DEFAULT_CHECKPOINT = REPO_ROOT / "stage1" / "checkpoint" / "vggt_1b_model.pt"


def disable_torch_compile() -> None:
    """Disable torch.compile decorators before importing VGGT-X modules.

    Some environments have a PyTorch/Jinja2/Inductor mismatch that fails during
    module import, before inference starts. Replacing torch.compile with an
    identity decorator keeps VGGT-X usable without editing the submodule.
    """

    def identity_compile(fn=None, *args, **kwargs):
        if fn is None:
            return lambda real_fn: real_fn
        return fn

    torch.compile = identity_compile


def add_vggt_x_to_path() -> None:
    """Make vggt-x local modules importable without installing the submodule."""
    if not VGGT_X_ROOT.is_dir():
        raise FileNotFoundError(f"Cannot find VGGT-X submodule: {VGGT_X_ROOT}")
    if str(VGGT_X_ROOT) not in sys.path:
        sys.path.insert(0, str(VGGT_X_ROOT))


def load_state_dict(checkpoint_path: Path) -> dict:
    """Load a local checkpoint and unwrap common checkpoint container formats."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Pass --checkpoint or place the model at stage1/checkpoint/vggt_1b_model.pt."
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state_dict and isinstance(state_dict[key], dict):
                return state_dict[key]
    return state_dict


def patch_demo_colmap(checkpoint_path: Path, enable_torch_compile: bool):
    """Import VGGT-X demo_colmap and patch its run_VGGT function."""
    add_vggt_x_to_path()
    if not enable_torch_compile:
        disable_torch_compile()

    import demo_colmap  # type: ignore

    def run_vggt_with_local_checkpoint(images, device, dtype, chunk_size):
        # Keep the original VGGT-X inference path, only replacing remote weight loading.
        model = demo_colmap.VGGT(chunk_size=chunk_size)
        print(f"Loading checkpoint from local file: {checkpoint_path}")
        model.load_state_dict(load_state_dict(checkpoint_path))
        model.eval()
        model = model.to(device).to(dtype)
        model.track_head = None
        print("Model loaded")

        with torch.no_grad():
            predictions = model(images.to(device, dtype), verbose=True)
            extrinsic, intrinsic = demo_colmap.pose_encoding_to_extri_intri(
                predictions["pose_enc"], images.shape[-2:]
            )
            extrinsic = extrinsic.squeeze(0).cpu().numpy()
            intrinsic = intrinsic.squeeze(0).cpu().numpy()
            depth_map = predictions["depth"].squeeze(0).cpu().numpy()
            depth_conf = predictions["depth_conf"].squeeze(0).cpu().numpy()

        return extrinsic, intrinsic, depth_map, depth_conf

    demo_colmap.run_VGGT = run_vggt_with_local_checkpoint
    return demo_colmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT-X demo_colmap with a local checkpoint from the main project."
    )
    parser.add_argument("--scene_dir", required=True, help="Scene directory containing an images/ folder.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Local VGGT checkpoint path. Default: stage1/checkpoint/vggt_1b_model.pt",
    )
    parser.add_argument("--post_fix", default="_vggt_x", help="Postfix for the output folder.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_ga", action="store_true", help="Apply VGGT-X global alignment.")
    parser.add_argument("--save_depth", action="store_true", help="Save estimated depth/confidence maps.")
    parser.add_argument("--chunk_size", type=int, default=256, help="VGGT frame chunk size.")
    parser.add_argument("--total_frame_num", type=int, default=None, help="Use only the first N images.")
    parser.add_argument("--max_query_pts", type=int, default=None, help="Maximum XFeat query points per pair.")
    parser.add_argument("--max_points_for_colmap", type=int, default=500000, help="Maximum COLMAP points.")
    parser.add_argument("--shared_camera", action="store_true", help="Use shared intrinsics during GA.")
    parser.add_argument(
        "--enable_torch_compile",
        action="store_true",
        help="Keep VGGT-X torch.compile enabled. Default disables it for better environment compatibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.scene_dir = os.path.abspath(args.scene_dir)
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()

    demo_colmap = patch_demo_colmap(checkpoint_path, args.enable_torch_compile)
    demo_colmap.demo_fn(args)


if __name__ == "__main__":
    main()
