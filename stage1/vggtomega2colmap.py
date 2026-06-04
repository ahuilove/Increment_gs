#!/usr/bin/env python3
"""
Convert VGGT-Omega predictions.npz to a COLMAP-style dataset.

The conversion follows the VGGT-X export idea: predicted camera poses,
intrinsics, dense depth and confidence are packed into cameras/images/points3D.
The 3D points are depth-unprojected points, not COLMAP-triangulated tracks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import Camera, Image, Point3D, rotmat2qvec, write_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    args = parse_args()

    predictions_path = Path(args.predictions).expanduser().resolve()
    images_dir = resolve_images_dir(args.images_dir, predictions_path)
    output_root = resolve_output_root(args.output, predictions_path)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    (output_root / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (output_root / "images").mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    predictions = load_predictions(predictions_path)
    num_images, pred_h, pred_w = prediction_shape(predictions)
    if len(image_paths) != num_images:
        raise ValueError(
            f"Image count ({len(image_paths)}) does not match prediction count ({num_images}). "
            f"images_dir={images_dir}"
        )

    preprocess = build_preprocess_geometry(
        image_paths=image_paths,
        pred_hw=(pred_h, pred_w),
        mode=args.preprocess_mode,
        image_resolution=args.image_resolution,
        patch_size=args.patch_size,
    )

    selected = select_points(
        predictions=predictions,
        max_points=args.max_points,
        conf_percentile=args.conf_percentile,
        sampling=args.sampling,
        seed=args.seed,
    )

    cameras, images, points3D = build_colmap_model(
        predictions=predictions,
        selected=selected,
        image_paths=image_paths,
        preprocess=preprocess,
        camera_model=args.camera_model,
    )

    write_model(cameras, images, points3D, str(output_root / "sparse" / "0"), ext=args.output_format)
    link_or_copy_images(image_paths, output_root / "images", args.copy_mode)
    if args.export_depths:
        write_depth_supervision(
            predictions=predictions,
            image_paths=image_paths,
            preprocess=preprocess,
            output_root=output_root,
            depths_name=args.depths_name,
            png_scale=args.depth_png_scale,
            invalid_value=args.depth_invalid_value,
            normalization=args.depth_normalization,
            scale_percentile=args.depth_scale_percentile,
        )
    write_summary(
        output_root=output_root,
        predictions_path=predictions_path,
        images_dir=images_dir,
        args=args,
        num_images=num_images,
        pred_hw=(pred_h, pred_w),
        num_points=len(points3D),
    )

    print(f"Wrote COLMAP dataset: {output_root}")
    print(f"Sparse model: {output_root / 'sparse' / '0'}")
    print(f"Images: {len(images)}")
    print(f"Points3D: {len(points3D)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VGGT-Omega predictions.npz to COLMAP format.")
    parser.add_argument("--predictions", required=True, help="Path to VGGT-Omega predictions.npz.")
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Input image directory. Defaults to predictions parent / images.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output COLMAP dataset root. Defaults to predictions parent / colmap.",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=["balanced", "max_size"],
        default="balanced",
        help="VGGT-Omega preprocessing mode used when predictions were created.",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=512,
        help="VGGT-Omega image_resolution used when predictions were created.",
    )
    parser.add_argument("--patch-size", type=int, default=16, help="VGGT-Omega patch size.")
    parser.add_argument("--max-points", type=int, default=500000, help="Maximum exported 3D points.")
    parser.add_argument(
        "--conf-percentile",
        type=float,
        default=0.5,
        help="Keep points with depth_conf at or above this percentile before max-points sampling.",
    )
    parser.add_argument(
        "--sampling",
        choices=["confidence", "random"],
        default="confidence",
        help="How to limit points if more than --max-points survive the confidence filter.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --sampling random.")
    parser.add_argument("--camera-model", choices=["PINHOLE", "SIMPLE_PINHOLE"], default="PINHOLE")
    parser.add_argument("--output-format", choices=[".bin", ".txt"], default=".bin")
    parser.add_argument("--copy-mode", choices=["symlink", "hardlink", "copy", "none"], default="symlink")
    parser.add_argument(
        "--export-depths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export inverse-depth PNGs and sparse/0/depth_params.json for train.py depth supervision.",
    )
    parser.add_argument(
        "--depths-name",
        default="depth",
        help="Depth folder name under output root. Pass the same value to train.py --depths.",
    )
    parser.add_argument(
        "--depth-png-scale",
        type=float,
        default=65536.0,
        help="Scale applied before saving inverse-depth PNGs. train.py divides non-synthetic depths by 2**16.",
    )
    parser.add_argument(
        "--depth-invalid-value",
        type=int,
        default=0,
        help="Uint16 value for invalid or non-positive depth pixels.",
    )
    parser.add_argument(
        "--depth-normalization",
        choices=["global_scale", "raw"],
        default="global_scale",
        help=(
            "global_scale stores normalized inverse depth PNGs and restores metric scale through depth_params.json. "
            "raw stores inverse_depth * --depth-png-scale directly and may saturate uint16."
        ),
    )
    parser.add_argument(
        "--depth-scale-percentile",
        type=float,
        default=99.9,
        help="Robust percentile used as global inverse-depth scale when --depth-normalization global_scale.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it exists.")
    return parser.parse_args()


def resolve_images_dir(images_dir_arg: str | None, predictions_path: Path) -> Path:
    candidates = []
    if images_dir_arg:
        candidates.append(Path(images_dir_arg).expanduser())
    candidates.append(predictions_path.parent / "images")
    candidates.append(predictions_path.parent.parent / "images")

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find images directory. Tried: {candidates}")


def resolve_output_root(output_arg: str | None, predictions_path: Path) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    return (predictions_path.parent / "colmap").resolve()


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        required = {"extrinsic", "intrinsic", "depth", "depth_conf"}
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")
        return {key: np.asarray(data[key]) for key in data.files}


def prediction_shape(predictions: dict[str, np.ndarray]) -> tuple[int, int, int]:
    extrinsic = squeeze_batch(predictions["extrinsic"])
    intrinsic = squeeze_batch(predictions["intrinsic"])
    depth = squeeze_batch(predictions["depth"])
    depth_conf = squeeze_batch(predictions["depth_conf"])

    if extrinsic.shape[-2:] != (3, 4):
        raise ValueError(f"Expected extrinsic shape [N,3,4], got {extrinsic.shape}")
    if intrinsic.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsic shape [N,3,3], got {intrinsic.shape}")
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth shape [N,H,W] or [N,H,W,1], got {predictions['depth'].shape}")
    if depth_conf.shape != depth.shape:
        raise ValueError(f"depth_conf shape {depth_conf.shape} does not match depth shape {depth.shape}")
    if len(extrinsic) != len(intrinsic) or len(extrinsic) != len(depth):
        raise ValueError("extrinsic, intrinsic and depth have inconsistent image counts.")
    return int(depth.shape[0]), int(depth.shape[1]), int(depth.shape[2])


def squeeze_batch(array: np.ndarray) -> np.ndarray:
    if array.ndim >= 1 and array.shape[0] == 1 and array.ndim in {4, 5}:
        return array[0]
    return array


def build_preprocess_geometry(
    image_paths: list[Path],
    pred_hw: tuple[int, int],
    mode: str,
    image_resolution: int,
    patch_size: int,
) -> list[dict[str, float]]:
    raw_geometries = []
    shapes = []
    for image_path in image_paths:
        with PILImage.open(image_path) as image:
            orig_w, orig_h = image.size
        crop_left, crop_top, crop_w, crop_h = supported_aspect_crop(orig_w, orig_h)
        aspect_ratio = crop_h / max(crop_w, 1)
        if mode == "balanced":
            target_h, target_w = balanced_target_shape(aspect_ratio, image_resolution, patch_size)
        else:
            target_h, target_w = max_size_target_shape(aspect_ratio, image_resolution, patch_size)
        shapes.append((target_h, target_w))
        raw_geometries.append(
            {
                "orig_w": float(orig_w),
                "orig_h": float(orig_h),
                "crop_left": float(crop_left),
                "crop_top": float(crop_top),
                "crop_w": float(crop_w),
                "crop_h": float(crop_h),
                "target_w": float(target_w),
                "target_h": float(target_h),
            }
        )

    common_h = max(h for h, _ in shapes)
    common_w = max(w for _, w in shapes)
    pred_h, pred_w = pred_hw
    if (common_h, common_w) != (pred_h, pred_w):
        raise ValueError(
            "Recomputed VGGT-Omega preprocessed shape does not match predictions. "
            f"computed={(common_h, common_w)}, predictions={(pred_h, pred_w)}. "
            "Check --preprocess-mode, --image-resolution and --patch-size."
        )

    for geom, (target_h, target_w) in zip(raw_geometries, shapes):
        geom["pad_left"] = float((common_w - target_w) // 2)
        geom["pad_top"] = float((common_h - target_h) // 2)
        geom["scale_x"] = float(target_w / geom["crop_w"])
        geom["scale_y"] = float(target_h / geom["crop_h"])
    return raw_geometries


def supported_aspect_crop(width: int, height: int) -> tuple[int, int, int, int]:
    aspect_ratio = height / max(width, 1)
    if aspect_ratio < 0.5:
        crop_width = min(width, max(1, int(round(height / 0.5))))
        left = max((width - crop_width) // 2, 0)
        return left, 0, crop_width, height
    if aspect_ratio > 2.0:
        crop_height = min(height, max(1, int(round(width * 2.0))))
        top = max((height - crop_height) // 2, 0)
        return 0, top, width, crop_height
    return 0, 0, width, height


def balanced_target_shape(aspect_ratio: float, image_resolution: int, patch_size: int) -> tuple[int, int]:
    token_number = (image_resolution // patch_size) ** 2
    w_patches = np.sqrt(token_number / aspect_ratio)
    h_patches = token_number / w_patches
    w_patches = max(1, int(np.round(w_patches)))
    h_patches = max(1, int(np.round(h_patches)))
    return h_patches * patch_size, w_patches * patch_size


def max_size_target_shape(aspect_ratio: float, image_resolution: int, patch_size: int) -> tuple[int, int]:
    if aspect_ratio >= 1.0:
        height = image_resolution
        width = round_to_patch_multiple(image_resolution / aspect_ratio, patch_size)
    else:
        width = image_resolution
        height = round_to_patch_multiple(image_resolution * aspect_ratio, patch_size)
    return height, width


def round_to_patch_multiple(value: float, patch_size: int) -> int:
    return max(patch_size, int(np.round(float(value) / patch_size)) * patch_size)


def select_points(
    predictions: dict[str, np.ndarray],
    max_points: int,
    conf_percentile: float,
    sampling: str,
    seed: int,
) -> np.ndarray:
    depth = get_depth(predictions)
    depth_conf = squeeze_batch(predictions["depth_conf"]).astype(np.float64)
    world_points = get_world_points(predictions)

    finite = np.isfinite(world_points).all(axis=-1)
    valid = finite & np.isfinite(depth_conf) & np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise ValueError("No valid positive-depth points found in predictions.")

    threshold = np.percentile(depth_conf[valid], conf_percentile)
    valid &= depth_conf >= threshold
    flat_valid = np.flatnonzero(valid.reshape(-1))
    if flat_valid.size == 0:
        raise ValueError("No points survived the confidence threshold.")

    if max_points > 0 and flat_valid.size > max_points:
        if sampling == "random":
            rng = np.random.default_rng(seed)
            flat_valid = rng.choice(flat_valid, size=max_points, replace=False)
        else:
            conf_flat = depth_conf.reshape(-1)
            local = np.argpartition(conf_flat[flat_valid], -max_points)[-max_points:]
            flat_valid = flat_valid[local]

    return np.sort(flat_valid.astype(np.int64))


def get_depth(predictions: dict[str, np.ndarray]) -> np.ndarray:
    depth = squeeze_batch(predictions["depth"]).astype(np.float64)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    return depth


def get_world_points(predictions: dict[str, np.ndarray]) -> np.ndarray:
    if "world_points_from_depth" in predictions:
        return squeeze_batch(predictions["world_points_from_depth"]).astype(np.float64)

    depth = get_depth(predictions)
    extrinsic = squeeze_batch(predictions["extrinsic"]).astype(np.float64)
    intrinsic = squeeze_batch(predictions["intrinsic"]).astype(np.float64)
    return unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)


def unproject_depth_map_to_point_map(depth: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    num_frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    camera_points = np.stack(((x - cx) / fx * depth, (y - cy) / fy * depth, depth), axis=-1)

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum("sij,shwj->shwi", np.swapaxes(rotation, 1, 2), camera_points - translation[:, None, None, :])


def build_colmap_model(
    predictions: dict[str, np.ndarray],
    selected: np.ndarray,
    image_paths: list[Path],
    preprocess: list[dict[str, float]],
    camera_model: str,
) -> tuple[dict[int, Camera], dict[int, Image], dict[int, Point3D]]:
    extrinsic = squeeze_batch(predictions["extrinsic"]).astype(np.float64)
    intrinsic = squeeze_batch(predictions["intrinsic"]).astype(np.float64)
    depth = get_depth(predictions)
    num_images, pred_h, pred_w = depth.shape
    world_points = get_world_points(predictions)
    colors = get_prediction_colors(predictions, selected, (num_images, pred_h, pred_w))

    frame_idx, y_idx, x_idx = np.unravel_index(selected, (num_images, pred_h, pred_w))

    cameras = {}
    images = {}
    points3D = {}
    image_xys: list[list[np.ndarray]] = [[] for _ in range(num_images)]
    image_point_ids: list[list[int]] = [[] for _ in range(num_images)]

    for point_offset, (flat_idx, fidx, y, x) in enumerate(zip(selected, frame_idx, y_idx, x_idx), start=1):
        xyz = world_points.reshape(-1, 3)[flat_idx].astype(np.float64)
        xy_orig = processed_xy_to_original(np.array([float(x), float(y)], dtype=np.float64), preprocess[int(fidx)])
        point2d_idx = len(image_xys[int(fidx)])
        image_xys[int(fidx)].append(xy_orig)
        image_point_ids[int(fidx)].append(point_offset)
        points3D[point_offset] = Point3D(
            id=point_offset,
            xyz=xyz,
            rgb=colors[point_offset - 1],
            error=0.0,
            image_ids=np.array([int(fidx) + 1], dtype=np.int32),
            point2D_idxs=np.array([point2d_idx], dtype=np.int32),
        )

    for idx, image_path in enumerate(image_paths):
        image_id = idx + 1
        camera_id = image_id
        k_orig = processed_k_to_original(intrinsic[idx], preprocess[idx])
        width = int(round(preprocess[idx]["orig_w"]))
        height = int(round(preprocess[idx]["orig_h"]))
        cameras[camera_id] = make_camera(camera_id, camera_model, width, height, k_orig)

        rotation = extrinsic[idx, :3, :3]
        qvec = rotmat2qvec(rotation).astype(np.float64)
        tvec = extrinsic[idx, :3, 3].astype(np.float64)
        xys = np.asarray(image_xys[idx], dtype=np.float64).reshape(-1, 2)
        point_ids = np.asarray(image_point_ids[idx], dtype=np.int64)
        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_path.name,
            xys=xys,
            point3D_ids=point_ids,
        )

    return cameras, images, points3D


def get_prediction_colors(
    predictions: dict[str, np.ndarray],
    selected: np.ndarray,
    shape_nhw: tuple[int, int, int],
) -> np.ndarray:
    num_images, height, width = shape_nhw
    if "images" not in predictions:
        return np.full((len(selected), 3), 128, dtype=np.uint8)

    images = squeeze_batch(predictions["images"])
    if images.shape[:2] == (num_images, 3):
        images = np.moveaxis(images, 1, -1)
    if images.shape[:3] != (num_images, height, width):
        return np.full((len(selected), 3), 128, dtype=np.uint8)

    colors = images.reshape(-1, 3)[selected]
    if np.issubdtype(colors.dtype, np.floating):
        colors = np.clip(colors, 0.0, 1.0) * 255.0
    return np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def processed_k_to_original(k: np.ndarray, geom: dict[str, float]) -> np.ndarray:
    out = np.asarray(k, dtype=np.float64).copy()
    scale_x = geom["scale_x"]
    scale_y = geom["scale_y"]
    out[0, 0] /= scale_x
    out[1, 1] /= scale_y
    out[0, 2] = (out[0, 2] - geom["pad_left"]) / scale_x + geom["crop_left"]
    out[1, 2] = (out[1, 2] - geom["pad_top"]) / scale_y + geom["crop_top"]
    return out


def processed_xy_to_original(xy: np.ndarray, geom: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            (xy[0] - geom["pad_left"]) / geom["scale_x"] + geom["crop_left"],
            (xy[1] - geom["pad_top"]) / geom["scale_y"] + geom["crop_top"],
        ],
        dtype=np.float64,
    )


def make_camera(camera_id: int, model: str, width: int, height: int, k: np.ndarray) -> Camera:
    if model == "PINHOLE":
        params = np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], dtype=np.float64)
    elif model == "SIMPLE_PINHOLE":
        focal = 0.5 * (k[0, 0] + k[1, 1])
        params = np.array([focal, k[0, 2], k[1, 2]], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported camera model: {model}")
    return Camera(id=camera_id, model=model, width=width, height=height, params=params)


def link_or_copy_images(image_paths: list[Path], output_images_dir: Path, copy_mode: str) -> None:
    if copy_mode == "none":
        return
    for src in image_paths:
        dst = output_images_dir / src.name
        if copy_mode == "symlink":
            os.symlink(src.resolve(), dst)
        elif copy_mode == "hardlink":
            os.link(src, dst)
        elif copy_mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unknown copy mode: {copy_mode}")


def write_summary(
    output_root: Path,
    predictions_path: Path,
    images_dir: Path,
    args: argparse.Namespace,
    num_images: int,
    pred_hw: tuple[int, int],
    num_points: int,
) -> None:
    summary = {
        "predictions": str(predictions_path),
        "images_dir": str(images_dir),
        "num_images": num_images,
        "prediction_height": pred_hw[0],
        "prediction_width": pred_hw[1],
        "num_points3D": num_points,
        "preprocess_mode": args.preprocess_mode,
        "image_resolution": args.image_resolution,
        "patch_size": args.patch_size,
        "conf_percentile": args.conf_percentile,
        "max_points": args.max_points,
        "sampling": args.sampling,
        "camera_model": args.camera_model,
        "copy_mode": args.copy_mode,
        "export_depths": args.export_depths,
        "depths_name": args.depths_name,
        "depth_png_scale": args.depth_png_scale,
        "depth_normalization": args.depth_normalization,
        "depth_scale_percentile": args.depth_scale_percentile,
        "note": "Points are VGGT-Omega depth-unprojected points with single-view observations, not COLMAP SfM tracks.",
    }
    with (output_root / "conversion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def write_depth_supervision(
    predictions: dict[str, np.ndarray],
    image_paths: list[Path],
    preprocess: list[dict[str, float]],
    output_root: Path,
    depths_name: str,
    png_scale: float,
    invalid_value: int,
    normalization: str,
    scale_percentile: float,
) -> None:
    """Write inverse-depth PNGs in the format consumed by this repo's train.py.

    The renderer returns inverse depth, and utils/camera_utils.py loads each
    non-synthetic depth PNG as uint16 / 2**16. Therefore we save:

        png = round((1 / VGGT_depth_in_original_camera_scale) * png_scale)

    With global_scale normalization, depth_params.json restores the normalized
    PNG values to VGGT-Omega's inverse-depth scale during training.
    """
    depth = get_depth(predictions)
    depth_dir = output_root / depths_name
    depth_dir.mkdir(parents=True, exist_ok=True)

    global_scale = 1.0
    if normalization == "global_scale":
        all_valid_inv_depths = []
        for idx in range(len(image_paths)):
            inv_depth = processed_depth_to_original_inv_depth(depth[idx], preprocess[idx])
            valid = np.isfinite(inv_depth) & (inv_depth > 0)
            if np.any(valid):
                all_valid_inv_depths.append(inv_depth[valid])
        if not all_valid_inv_depths:
            raise ValueError("No valid inverse-depth pixels found for depth export.")
        all_valid_inv_depths = np.concatenate(all_valid_inv_depths)
        global_scale = float(np.percentile(all_valid_inv_depths, scale_percentile))
        if not np.isfinite(global_scale) or global_scale <= 1e-12:
            global_scale = float(np.max(all_valid_inv_depths))
        if not np.isfinite(global_scale) or global_scale <= 1e-12:
            raise ValueError("Could not compute a positive inverse-depth scale.")

    depth_params = {}
    invalid_value = int(np.clip(invalid_value, 0, np.iinfo(np.uint16).max))
    for idx, image_path in enumerate(image_paths):
        inv_depth = processed_depth_to_original_inv_depth(depth[idx], preprocess[idx])
        valid = np.isfinite(inv_depth) & (inv_depth > 0)
        encoded = np.full(inv_depth.shape, invalid_value, dtype=np.uint16)
        if normalization == "global_scale":
            stored_inv_depth = inv_depth[valid] / global_scale
            depth_params[image_path.stem] = {"scale": global_scale, "offset": 0.0}
        else:
            stored_inv_depth = inv_depth[valid]
            depth_params[image_path.stem] = {"scale": 1.0, "offset": 0.0}

        encoded_values = np.clip(np.rint(stored_inv_depth * png_scale), 1, np.iinfo(np.uint16).max)
        encoded[valid] = encoded_values.astype(np.uint16)

        output_name = f"{image_path.stem}.png"
        cv2.imwrite(str(depth_dir / output_name), encoded)

    with (output_root / "sparse" / "0" / "depth_params.json").open("w", encoding="utf-8") as f:
        json.dump(depth_params, f, indent=2)


def processed_depth_to_original_inv_depth(depth: np.ndarray, geom: dict[str, float]) -> np.ndarray:
    target_w = int(round(geom["target_w"]))
    target_h = int(round(geom["target_h"]))
    pad_left = int(round(geom["pad_left"]))
    pad_top = int(round(geom["pad_top"]))
    crop_w = int(round(geom["crop_w"]))
    crop_h = int(round(geom["crop_h"]))
    orig_w = int(round(geom["orig_w"]))
    orig_h = int(round(geom["orig_h"]))
    crop_left = int(round(geom["crop_left"]))
    crop_top = int(round(geom["crop_top"]))

    processed_crop = depth[pad_top : pad_top + target_h, pad_left : pad_left + target_w]
    inv_processed = np.zeros_like(processed_crop, dtype=np.float32)
    valid = np.isfinite(processed_crop) & (processed_crop > 1e-12)
    inv_processed[valid] = (1.0 / processed_crop[valid]).astype(np.float32)

    inv_crop = cv2.resize(inv_processed, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    inv_original = np.zeros((orig_h, orig_w), dtype=np.float32)
    inv_original[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w] = inv_crop
    return inv_original


if __name__ == "__main__":
    main()
