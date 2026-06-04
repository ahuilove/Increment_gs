# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


'''
食用方法：
用图片文件夹运行：

  python demo_vggt_omega.py \
    --images dataset \
    --checkpoint checkpoint/vggt_omega_1b_512.pt \
    --output-dir demo_outputs/my_cli_run \
    --image-resolution 512

  用视频运行：

  python demo_vggt_omega.py \
    --video examples/forest_road.mp4 \
    --checkpoint checkpoint/vggt_omega_1b_512.pt \
    --output-dir demo_outputs/forest_cli_run \
    --video-sample-fps 1.0 \
    --image-resolution 512

  输出目录结构类似：

  demo_outputs/my_cli_run/
    images/
    predictions.npz
    scene_conf50.0_blackFalse_whiteFalse_camTrue_skyFalse_max1000k.glb
'''

import argparse
import gc
import glob
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
VGGT_OMEGA_ROOT = REPO_ROOT / "submodules" / "vggt-omega"

if str(VGGT_OMEGA_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_OMEGA_ROOT))

from visual_util import predictions_to_glb
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    args = parse_args()

    target_dir = prepare_inputs(
        input_video=args.video,
        input_images_dir=args.images,
        output_dir=args.output_dir,
        video_sample_fps=args.video_sample_fps,
        overwrite=args.overwrite,
    )

    model = load_model(args.checkpoint)
    predictions = run_model(target_dir, model, args.image_resolution, mode=args.preprocess_mode)

    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)
    print(f"Saved predictions: {prediction_save_path}")

    glbfile = glb_path(
        target_dir,
        args.conf_thres,
        args.mask_black_bg,
        args.mask_white_bg,
        args.show_cam,
        args.mask_sky,
        args.max_points_k,
    )
    scene = predictions_to_glb(
        predictions,
        conf_thres=args.conf_thres,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        show_cam=args.show_cam,
        mask_sky=args.mask_sky,
        target_dir=target_dir,
        max_points=int(args.max_points_k * 1000),
        filter_depth_edges=not args.no_filter_depth_edges,
        depth_edge_rtol=args.depth_edge_rtol,
    )
    scene.export(file_obj=glbfile)
    print(f"Saved GLB: {glbfile}")

    del predictions
    gc.collect()
    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Command-line VGGT-Omega reconstruction demo.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", help="Input video path.")
    input_group.add_argument("--images", help="Input image folder.")

    parser.add_argument("--checkpoint", default="stage1/checkpoint/vggt_omega_1b_512.pt", help="Local VGGT-Omega checkpoint path.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to demo_outputs/input_images_TIMESTAMP.")
    parser.add_argument("--overwrite", action="store_true", help="Allow reusing and overwriting an existing output dir.")
    parser.add_argument("--image-resolution", type=int, default=512, help="Input image resolution. Default: 512.")
    parser.add_argument(
        "--preprocess-mode",
        choices=["balanced", "max_size"],
        default="balanced",
        help="Image preprocessing mode used by load_and_preprocess_images.",
    )
    parser.add_argument("--video-sample-fps", type=float, default=1.0, help="FPS used when sampling video frames.")
    parser.add_argument("--conf-thres", type=float, default=50.0, help="Point-cloud confidence percentile threshold.")
    parser.add_argument("--max-points-k", type=int, default=1000, help="Maximum GLB points in thousands.")
    parser.add_argument("--mask-black-bg", action="store_true", help="Filter near-black background points.")
    parser.add_argument("--mask-white-bg", action="store_true", help="Filter near-white background points.")
    parser.add_argument("--mask-sky", action="store_true", help="Filter sky using the optional skyseg model.")
    parser.add_argument("--show-cam", action=argparse.BooleanOptionalAction, default=True, help="Show camera frustums.")
    parser.add_argument("--no-filter-depth-edges", action="store_true", help="Disable depth-edge point filtering.")
    parser.add_argument("--depth-edge-rtol", type=float, default=0.03, help="Relative depth-edge threshold.")
    return parser.parse_args()


def prepare_inputs(
    input_video: str | None,
    input_images_dir: str | None,
    output_dir: str | None,
    video_sample_fps: float,
    overwrite: bool,
) -> str:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target_dir = os.path.join("demo_outputs", f"input_images_{timestamp}")
    else:
        target_dir = output_dir

    target_dir_images = os.path.join(target_dir, "images")
    if os.path.exists(target_dir):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {target_dir}. Use --overwrite to reuse it.")
        if os.path.exists(target_dir_images):
            shutil.rmtree(target_dir_images)
    os.makedirs(target_dir_images, exist_ok=True)

    if input_images_dir is not None:
        image_paths = copy_input_images(input_images_dir, target_dir_images)
    else:
        image_paths = extract_video_frames(input_video, target_dir_images, video_sample_fps)

    if not image_paths:
        raise ValueError("No input frames/images were found.")

    print(f"Prepared {len(image_paths)} images in {target_dir_images}")
    return target_dir


def copy_input_images(input_images_dir: str, target_dir_images: str) -> list[str]:
    src_dir = Path(input_images_dir)
    if not src_dir.is_dir():
        raise NotADirectoryError(input_images_dir)

    image_paths = []
    for src_path in sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
        dst_path = Path(target_dir_images) / src_path.name
        shutil.copy2(src_path, dst_path)
        image_paths.append(str(dst_path))
    return image_paths


def extract_video_frames(input_video: str, target_dir_images: str, video_sample_fps: float) -> list[str]:
    if input_video is None:
        raise ValueError("--video is required when no --images directory is provided")
    if not os.path.isfile(input_video):
        raise FileNotFoundError(input_video)

    video = cv2.VideoCapture(input_video)
    if not video.isOpened():
        raise ValueError(f"Failed to open video: {input_video}")

    fps = video.get(cv2.CAP_PROP_FPS)
    video_sample_fps = max(float(video_sample_fps), 0.1)
    frame_interval = max(int(round((fps if fps and fps > 0 else 1) / video_sample_fps)), 1)

    image_paths = []
    frame_idx = 0
    saved_idx = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            image_path = os.path.join(target_dir_images, f"{saved_idx:06}.png")
            cv2.imwrite(image_path, frame)
            image_paths.append(image_path)
            saved_idx += 1
        frame_idx += 1
    video.release()
    return image_paths


def load_model(checkpoint_path: str) -> VGGTOmega:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run VGGT-Omega.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to("cuda")


def run_model(target_dir: str, model: VGGTOmega, image_resolution: int, mode: str) -> dict:
    print(f"Processing images from {target_dir}")

    image_names = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    if len(image_names) == 0:
        raise ValueError("No images found.")

    images = load_and_preprocess_images(image_names, mode=mode, image_resolution=image_resolution).to("cuda")
    print(f"Preprocessed images shape: {tuple(images.shape)}")

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value

    predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )

    torch.cuda.empty_cache()
    return predictions_np


def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def glb_path(
    target_dir: str,
    conf_thres: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
    show_cam: bool,
    mask_sky: bool,
    max_points_k: int,
) -> str:
    return os.path.join(
        target_dir,
        f"scene_conf{conf_thres}_black{mask_black_bg}_white{mask_white_bg}_"
        f"cam{show_cam}_sky{mask_sky}_max{int(max_points_k)}k.glb",
    )


if __name__ == "__main__":
    main()
