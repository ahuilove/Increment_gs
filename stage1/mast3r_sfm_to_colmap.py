#!/usr/bin/env python3
"""Export a saved MASt3R-SfM scene.pt to a COLMAP sparse model.

第一版导出策略是 per_view_points：
  - cameras.bin/images.bin 来自 MASt3R-SfM 的 intrinsics 和 cam2world；
  - 每张图的 sparse_pts3d 独立写成 COLMAP Point3D；
  - 每个 Point3D 只有一个 track 观测，即它所属的那张图像。

这样生成的是合法 COLMAP sparse/0，适合先打通后续 3DGS/loader 流程。
它不是 COLMAP 那种多视角 track-aware triangulation；后续可以再从 MASt3R
cache correspondences 构建多视角 tracks。
"""

import argparse
import csv
import json
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import (  # noqa: E402
    Camera,
    Image,
    Point3D,
    rotmat2qvec,
    write_model,
)


def load_scene(scene_path):
    scene_path = Path(scene_path).expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"Cannot find scene.pt: {scene_path}")
    try:
        return torch.load(scene_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(scene_path, map_location="cpu")


def to_numpy(value, dtype=np.float64):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def mast3r_processed_geometry(width, height, image_size, patch_size=16, square_ok=False):
    """复现 dust3r.utils.image.load_images 的 resize/crop 几何。

    返回：
      processed_width, processed_height, resize_scale, crop_left, crop_top

    MASt3R 的 K 在 crop 后图像坐标中；映射回原图需要：
      x_orig = (x_proc + crop_left) / resize_scale
    """
    scale = float(image_size) / float(max(width, height))
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    cx = resized_width // 2
    cy = resized_height // 2
    halfw = ((2 * cx) // patch_size) * patch_size / 2.0
    halfh = ((2 * cy) // patch_size) * patch_size / 2.0
    if not square_ok and resized_width == resized_height:
        halfh = 3.0 * halfw / 4.0

    crop_left = cx - halfw
    crop_top = cy - halfh
    processed_width = int(round(2.0 * halfw))
    processed_height = int(round(2.0 * halfh))
    return processed_width, processed_height, scale, crop_left, crop_top


def scale_intrinsics_to_original(K_proc, width, height, image_size):
    """把 MASt3R 输入图坐标下的 K 转成原始图片坐标下的 PINHOLE 参数。"""
    _, _, scale, crop_left, crop_top = mast3r_processed_geometry(width, height, image_size)
    fx = float(K_proc[0, 0]) / scale
    fy = float(K_proc[1, 1]) / scale
    cx = (float(K_proc[0, 2]) + crop_left) / scale
    cy = (float(K_proc[1, 2]) + crop_top) / scale
    return np.array([fx, fy, cx, cy], dtype=np.float64)


def cam2world_to_colmap_qt(cam2world):
    """MASt3R cam2world -> COLMAP world-to-camera qvec/tvec。"""
    cam2world = np.asarray(cam2world, dtype=np.float64)
    world_to_cam = np.linalg.inv(cam2world)
    rotation = world_to_cam[:3, :3]
    tvec = world_to_cam[:3, 3]
    qvec = rotmat2qvec(rotation)
    return qvec.astype(np.float64), tvec.astype(np.float64)


def image_size(image_path):
    with PILImage.open(image_path) as image:
        return image.size


def copy_or_link_images(image_paths, image_names, dst_images_dir, mode):
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    for src_path, image_name in zip(image_paths, image_names):
        src = Path(src_path)
        dst = dst_images_dir / image_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "symlink":
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src.resolve(), dst)
        elif mode == "hardlink":
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError as exc:
                if exc.errno != 18:  # EXDEV: cross-device link
                    raise
                shutil.copy2(src, dst)
        elif mode == "none":
            continue
        else:
            raise ValueError(f"Unknown copy mode: {mode}")


def make_placeholder_xys(num_points, width, height):
    """为 per-view points 生成稳定的 2D 坐标。

    当前 scene.pt 没有保存 MASt3R sparse anchors 的像素坐标，因此这里生成
    一组覆盖图像范围的占位坐标。COLMAP 模型是合法的；后续 track-aware 版本
    应改为使用真实 correspondences/anchors。
    """
    if num_points <= 0:
        return np.empty((0, 2), dtype=np.float64)
    cols = int(np.ceil(np.sqrt(num_points * max(width, 1) / max(height, 1))))
    rows = int(np.ceil(num_points / cols))
    xs = np.linspace(0.5, max(width - 0.5, 0.5), cols, dtype=np.float64)
    ys = np.linspace(0.5, max(height - 0.5, 0.5), rows, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)[:num_points]


def normalize_colors(colors, count):
    if colors is None:
        return np.full((count, 3), 255, dtype=np.uint8)
    colors = np.asarray(colors)
    if colors.size == 0:
        return np.full((count, 3), 255, dtype=np.uint8)
    if colors.dtype.kind == "f":
        if colors.max(initial=0.0) <= 1.0:
            colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    if len(colors) != count:
        colors = np.resize(colors, (count, 3)).astype(np.uint8)
    return colors


def build_colmap_model(scene, images_dir, image_size_value, max_points_per_image):
    image_paths = [Path(p).expanduser().resolve() for p in scene["image_paths"]]
    image_names = list(scene["image_names"])
    intrinsics = to_numpy(scene["intrinsics"])
    cam2world = to_numpy(scene["cam2world"])
    sparse_pts3d = scene["sparse_pts3d"]
    sparse_colors = scene.get("sparse_colors", [None] * len(image_names))

    if not (len(image_names) == len(image_paths) == len(intrinsics) == len(cam2world)):
        raise ValueError("scene.pt has inconsistent image/intrinsic/pose counts.")

    cameras = OrderedDict()
    images = OrderedDict()
    points3D = OrderedDict()
    mapping_rows = []
    next_point_id = 1

    for idx, (image_name, image_path) in enumerate(zip(image_names, image_paths)):
        if not image_path.is_file():
            fallback = images_dir / image_name
            if fallback.is_file():
                image_path = fallback
            else:
                raise FileNotFoundError(f"Cannot find image: {image_path}")

        width, height = image_size(image_path)
        camera_id = idx + 1
        image_id = idx + 1
        camera_params = scale_intrinsics_to_original(
            intrinsics[idx], width, height, image_size_value
        )
        cameras[camera_id] = Camera(
            id=camera_id,
            model="PINHOLE",
            width=width,
            height=height,
            params=camera_params,
        )

        qvec, tvec = cam2world_to_colmap_qt(cam2world[idx])
        pts = to_numpy(sparse_pts3d[idx], dtype=np.float64)
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        colors = normalize_colors(sparse_colors[idx], len(valid))[valid]

        if max_points_per_image is not None and len(pts) > max_points_per_image:
            keep = np.linspace(0, len(pts) - 1, max_points_per_image).round().astype(np.int64)
            pts = pts[keep]
            colors = colors[keep]

        xys = make_placeholder_xys(len(pts), width, height)
        point_ids = np.arange(next_point_id, next_point_id + len(pts), dtype=np.int64)

        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,
            xys=xys,
            point3D_ids=point_ids,
        )

        for local_idx, (point_id, xyz, rgb) in enumerate(zip(point_ids, pts, colors)):
            points3D[int(point_id)] = Point3D(
                id=int(point_id),
                xyz=xyz.astype(np.float64),
                rgb=rgb.astype(np.uint8),
                error=0.0,
                image_ids=np.array([image_id], dtype=np.int32),
                point2D_idxs=np.array([local_idx], dtype=np.int32),
            )

        mapping_rows.append(
            {
                "image_id": image_id,
                "camera_id": camera_id,
                "image_name": image_name,
                "source_path": str(image_path),
                "width": width,
                "height": height,
                "num_points": len(pts),
            }
        )
        next_point_id += len(pts)

    return cameras, images, points3D, mapping_rows


def write_mapping(output_root, rows):
    with (output_root / "mast3r_to_colmap_mapping.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "camera_id",
                "image_name",
                "source_path",
                "width",
                "height",
                "num_points",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def export_scene(args):
    scene_path = Path(args.scene).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    images_dir = Path(args.images).expanduser().resolve() if args.images else None
    scene = load_scene(scene_path)

    if images_dir is None:
        images_dir = Path(scene.get("images_dir", scene_path.parent)).expanduser().resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Cannot find images dir: {images_dir}")

    image_size_value = args.image_size
    if image_size_value is None:
        image_size_value = int(scene.get("config", {}).get("image_size", 512))

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite.")
        shutil.rmtree(output_root)

    sparse_dir = output_root / "sparse" / "0"
    images_out = output_root / "images"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    cameras, images, points3D, mapping_rows = build_colmap_model(
        scene=scene,
        images_dir=images_dir,
        image_size_value=image_size_value,
        max_points_per_image=args.max_points_per_image,
    )
    write_model(cameras, images, points3D, str(sparse_dir), ext=args.output_format)
    resolved_image_paths = [row["source_path"] for row in mapping_rows]
    copy_or_link_images(resolved_image_paths, scene["image_names"], images_out, args.copy_mode)
    write_mapping(output_root, mapping_rows)

    summary = {
        "source_scene": str(scene_path),
        "images_dir": str(images_dir),
        "output_format": args.output_format,
        "export_mode": "per_view_points",
        "num_cameras": len(cameras),
        "num_images": len(images),
        "num_points3D": len(points3D),
        "max_points_per_image": args.max_points_per_image,
        "note": "Point3D tracks are single-view per-view points; this is not track-aware COLMAP triangulation.",
    }
    with (output_root / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Export MASt3R-SfM scene.pt to COLMAP sparse/0.")
    parser.add_argument("--scene", required=True, help="Input scene.pt from mast3r_sfm_reconstruct.py.")
    parser.add_argument("--output", required=True, help="Output COLMAP-like dataset root.")
    parser.add_argument("--images", default=None, help="Image folder. Default: images_dir stored in scene.pt.")
    parser.add_argument("--image_size", type=int, default=None, help="MASt3R input image size. Default: scene config or 512.")
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink", "symlink", "none"],
        default="symlink",
        help="How to populate output/images.",
    )
    parser.add_argument(
        "--output-format",
        choices=[".bin", ".txt"],
        default=".bin",
        help="COLMAP sparse model format.",
    )
    parser.add_argument(
        "--max_points_per_image",
        type=int,
        default=None,
        help="Optional cap for exported per-view points. Default exports all finite sparse points.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove output folder first if it exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_points_per_image is not None and args.max_points_per_image <= 0:
        raise ValueError("--max_points_per_image must be > 0 when set.")
    summary = export_scene(args)
    print(f"Wrote COLMAP sparse model to {Path(args.output).resolve() / 'sparse' / '0'}")
    print(
        f"cameras={summary['num_cameras']} images={summary['num_images']} "
        f"points3D={summary['num_points3D']}"
    )


if __name__ == "__main__":
    main()
