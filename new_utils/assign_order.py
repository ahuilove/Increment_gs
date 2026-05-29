#!/usr/bin/env python3
"""Create a COLMAP copy whose image names follow a spatial camera order."""

import argparse
import csv
import json
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import (
    Image,
    Point3D,
    qvec2rotmat,
    read_images_binary,
    read_images_text,
    read_model,
    write_images_binary,
    write_images_text,
    write_model,
)


def detect_sparse_format(sparse_dir):
    if (sparse_dir / "images.bin").is_file():
        return ".bin"
    if (sparse_dir / "images.txt").is_file():
        return ".txt"
    raise FileNotFoundError(f"Cannot find images.bin or images.txt in {sparse_dir}")


def read_images(sparse_dir, ext):
    if ext == ".bin":
        return read_images_binary(str(sparse_dir / "images.bin"))
    if ext == ".txt":
        return read_images_text(str(sparse_dir / "images.txt"))
    raise ValueError(f"Unsupported COLMAP extension: {ext}")


def write_images(images, sparse_dir, ext):
    if ext == ".bin":
        write_images_binary(images, str(sparse_dir / "images.bin"))
    elif ext == ".txt":
        write_images_text(images, str(sparse_dir / "images.txt"))
    else:
        raise ValueError(f"Unsupported COLMAP extension: {ext}")


def camera_center(image):
    # COLMAP stores world-to-camera as x_cam = R * x_world + t.
    return -qvec2rotmat(image.qvec).T @ image.tvec


def normalize_positions(centers, dims):
    coords = centers[:, :dims].astype(np.float64)
    coords = coords - coords.mean(axis=0, keepdims=True)
    scale = coords.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return coords / scale


def nearest_neighbor_order(centers, image_ids, dims):
    coords = normalize_positions(centers, dims)
    n_images = len(image_ids)
    if n_images == 0:
        return []

    corner_score = coords.sum(axis=1)
    start = int(np.lexsort((np.array(image_ids), corner_score))[0])

    visited = np.zeros(n_images, dtype=bool)
    order = []
    current = start
    for _ in range(n_images):
        order.append(current)
        visited[current] = True
        if len(order) == n_images:
            break

        remaining = np.flatnonzero(~visited)
        diff = coords[remaining] - coords[current]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        best = np.lexsort((np.array(image_ids)[remaining], dist2))[0]
        current = int(remaining[best])

    return [image_ids[i] for i in order]


def pca_serpentine_order(centers, image_ids):
    """Row-by-row order for aerial/grid captures, with alternating row direction."""
    xy = normalize_positions(centers, 2)
    _, _, vh = np.linalg.svd(xy, full_matrices=False)
    uv = xy @ vh.T

    v_sorted = np.sort(uv[:, 1])
    gaps = np.diff(v_sorted)
    positive = gaps[gaps > 1e-8]
    if len(positive) == 0:
        row_step = 1.0
    else:
        row_step = np.median(positive) * 4.0
    row_step = max(row_step, 1e-6)

    rows = np.floor((uv[:, 1] - uv[:, 1].min()) / row_step).astype(int)
    result = []
    for row_id in sorted(set(rows)):
        idx = np.flatnonzero(rows == row_id)
        if row_id % 2 == 0:
            row_order = idx[np.lexsort((np.array(image_ids)[idx], uv[idx, 0]))]
        else:
            row_order = idx[np.lexsort((np.array(image_ids)[idx], -uv[idx, 0]))]
        result.extend(row_order.tolist())
    return [image_ids[i] for i in result]


def make_order(images, method, dims):
    image_ids = list(images.keys())
    centers = np.stack([camera_center(images[image_id]) for image_id in image_ids], axis=0)

    if method == "nearest":
        return nearest_neighbor_order(centers, image_ids, dims)
    if method == "serpentine":
        return pca_serpentine_order(centers, image_ids)
    raise ValueError(f"Unknown order method: {method}")


def copy_file(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")


def copy_sparse_except_images(src_sparse, dst_sparse):
    dst_sparse.mkdir(parents=True, exist_ok=True)
    for src in src_sparse.iterdir():
        if src.name in {"images.bin", "images.txt"}:
            continue
        dst = dst_sparse / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def copy_sparse_extra_files(src_sparse, dst_sparse):
    dst_sparse.mkdir(parents=True, exist_ok=True)
    model_files = {
        "cameras.bin",
        "cameras.txt",
        "images.bin",
        "images.txt",
        "points3D.bin",
        "points3D.txt",
        "points3D.ply",
    }
    for src in src_sparse.iterdir():
        if src.name in model_files:
            continue
        dst = dst_sparse / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def ordered_image_name(index, old_name):
    suffix = Path(old_name).suffix
    return f"{index:06d}{suffix}"


def parse_mapping(mapping_path):
    entries = []
    seen_orders = set()
    seen_names = set()
    with Path(mapping_path).open("r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            if len(elems) < 2:
                raise ValueError(f"Invalid mapping line {line_no}: {line}")
            order = int(elems[0])
            name = elems[1]
            if order in seen_orders:
                raise ValueError(f"Duplicate order {order} in {mapping_path}")
            if name in seen_names:
                raise ValueError(f"Duplicate image name {name} in {mapping_path}")
            seen_orders.add(order)
            seen_names.add(name)
            entries.append({"order": order, "name": name})
    entries.sort(key=lambda item: item["order"])
    return entries


def remap_depth_scale_file(src_root, dst_root, name_map):
    src_file = src_root / "estimated_depth_scales.json"
    if not src_file.is_file():
        return
    dst_file = dst_root / "estimated_depth_scales.json"
    with src_file.open("r") as f:
        data = json.load(f)

    stem_map = {Path(k).stem: Path(v).stem for k, v in name_map.items()}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "image_name" in item:
                item["image_name"] = stem_map.get(item["image_name"], item["image_name"])
    elif isinstance(data, dict):
        data = {stem_map.get(k, k): v for k, v in data.items()}

    with dst_file.open("w") as f:
        json.dump(data, f, indent=2)


def copy_depth_folder(src_root, dst_root, name_map, folder_name, mode):
    src_dir = src_root / folder_name
    if not src_dir.is_dir():
        return
    dst_dir = dst_root / folder_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    for old_name, new_name in name_map.items():
        old_path = src_dir / f"{Path(old_name).stem}.png"
        if old_path.is_file():
            new_path = dst_dir / f"{Path(new_name).stem}.png"
            copy_file(old_path, new_path, mode)


def write_mapping(dst_root, ordered_ids, images, name_map):
    with (dst_root / "order_mapping.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order", "image_id", "old_name", "new_name", "center_x", "center_y", "center_z"])
        for order, image_id in enumerate(ordered_ids):
            image = images[image_id]
            center = camera_center(image)
            writer.writerow([order, image_id, image.name, name_map[image.name], *center.tolist()])


def build_output_from_mapping(src_root, dst_root, mapping_path, copy_mode, overwrite, copy_depths):
    src_images = src_root / "images"
    src_sparse = src_root / "sparse" / "0"
    dst_images = dst_root / "images"
    dst_sparse = dst_root / "sparse" / "0"

    if not src_images.is_dir():
        raise FileNotFoundError(f"Cannot find images folder: {src_images}")
    if not src_sparse.is_dir():
        raise FileNotFoundError(f"Cannot find sparse/0 folder: {src_sparse}")
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {dst_root}. Use --overwrite to replace it.")
        shutil.rmtree(dst_root)

    ext = detect_sparse_format(src_sparse)
    cameras, images, points3D = read_model(str(src_sparse), ext=ext)
    mapping = parse_mapping(mapping_path)
    images_by_name = {image.name: image for image in images.values()}

    ordered_ids = []
    for item in mapping:
        image = images_by_name.get(item["name"])
        if image is None:
            raise KeyError(f"image name {item['name']} from mapping is not in COLMAP images")
        ordered_ids.append(image.id)

    selected_ids = set(ordered_ids)
    selected_id_array = np.array(ordered_ids, dtype=np.int64)
    used_camera_ids = {images[image_id].camera_id for image_id in ordered_ids}
    selected_cameras = OrderedDict((cam_id, cameras[cam_id]) for cam_id in cameras if cam_id in used_camera_ids)

    referenced_point_ids = set()
    for image_id in ordered_ids:
        point_ids = images[image_id].point3D_ids
        referenced_point_ids.update(int(pid) for pid in point_ids if int(pid) >= 0)

    filtered_points = OrderedDict()
    for point_id, point in points3D.items():
        if point_id not in referenced_point_ids:
            continue
        keep_mask = np.isin(point.image_ids, selected_id_array)
        if not np.any(keep_mask):
            continue
        filtered_points[point_id] = Point3D(
            id=point.id,
            xyz=point.xyz,
            rgb=point.rgb,
            error=point.error,
            image_ids=point.image_ids[keep_mask],
            point2D_idxs=point.point2D_idxs[keep_mask],
        )

    kept_point_ids = set(filtered_points.keys())
    name_map = {}
    ordered_images = OrderedDict()

    dst_images.mkdir(parents=True, exist_ok=True)
    copy_sparse_extra_files(src_sparse, dst_sparse)

    for order, image_id in enumerate(ordered_ids):
        image = images[image_id]
        new_name = ordered_image_name(order, image.name)
        old_path = src_images / image.name
        new_path = dst_images / new_name
        if not old_path.is_file():
            raise FileNotFoundError(f"Image referenced by COLMAP is missing: {old_path}")
        copy_file(old_path, new_path, copy_mode)
        name_map[image.name] = new_name

        point3D_ids = np.array(
            [int(pid) if int(pid) in kept_point_ids else -1 for pid in image.point3D_ids],
            dtype=image.point3D_ids.dtype,
        )
        ordered_images[image_id] = Image(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=new_name,
            xys=image.xys,
            point3D_ids=point3D_ids,
        )

    write_model(selected_cameras, ordered_images, filtered_points, str(dst_sparse), ext=ext)
    write_mapping(dst_root, ordered_ids, images, name_map)
    shutil.copy2(mapping_path, dst_root / "mapping.txt")

    if copy_depths:
        copy_depth_folder(src_root, dst_root, name_map, "estimated_depths", copy_mode)
        remap_depth_scale_file(src_root, dst_root, name_map)

    return len(ordered_ids), len(filtered_points), ext


def build_output(src_root, dst_root, method, dims, copy_mode, overwrite, copy_depths):
    src_images = src_root / "images"
    src_sparse = src_root / "sparse" / "0"
    dst_images = dst_root / "images"
    dst_sparse = dst_root / "sparse" / "0"

    if not src_images.is_dir():
        raise FileNotFoundError(f"Cannot find images folder: {src_images}")
    if not src_sparse.is_dir():
        raise FileNotFoundError(f"Cannot find sparse/0 folder: {src_sparse}")
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {dst_root}. Use --overwrite to replace it.")
        shutil.rmtree(dst_root)

    ext = detect_sparse_format(src_sparse)
    images = read_images(src_sparse, ext)
    ordered_ids = make_order(images, method, dims)
    name_map = {}
    ordered_images = OrderedDict()

    dst_images.mkdir(parents=True, exist_ok=True)
    copy_sparse_except_images(src_sparse, dst_sparse)

    for order, image_id in enumerate(ordered_ids):
        image = images[image_id]
        new_name = ordered_image_name(order, image.name)
        old_path = src_images / image.name
        new_path = dst_images / new_name
        if not old_path.is_file():
            raise FileNotFoundError(f"Image referenced by COLMAP is missing: {old_path}")
        copy_file(old_path, new_path, copy_mode)
        name_map[image.name] = new_name
        ordered_images[image_id] = Image(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=new_name,
            xys=image.xys,
            point3D_ids=image.point3D_ids,
        )

    write_images(ordered_images, dst_sparse, ext)
    write_mapping(dst_root, ordered_ids, images, name_map)

    if copy_depths:
        copy_depth_folder(src_root, dst_root, name_map, "estimated_depths", copy_mode)
        remap_depth_scale_file(src_root, dst_root, name_map)

    return len(ordered_ids), ext


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename a COLMAP dataset so adjacent image names are spatially adjacent cameras."
    )
    parser.add_argument("--input", "-i", required=True, help="Input COLMAP root containing images and sparse/0.")
    parser.add_argument("--output", "-o", required=True, help="Output COLMAP root to create.")
    parser.add_argument("--mapping", "-m", default=None, help="Manual mapping.txt from colmap_order_gui.py. If set, only this ordered subset is exported.")
    parser.add_argument(
        "--method",
        choices=["nearest", "serpentine"],
        default="nearest",
        help="Ordering strategy. nearest is general; serpentine is useful for regular aerial grids.",
    )
    parser.add_argument("--dims", type=int, choices=[2, 3], default=2, help="Camera-center dimensions used by nearest.")
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to place images in the output folder.",
    )
    parser.add_argument("--copy-depths", action="store_true", help="Also remap estimated_depths and estimated_depth_scales.json if present.")
    parser.add_argument("--overwrite", action="store_true", help="Remove the output folder first if it exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mapping:
        count, point_count, ext = build_output_from_mapping(
            src_root=Path(args.input).resolve(),
            dst_root=Path(args.output).resolve(),
            mapping_path=Path(args.mapping).resolve(),
            copy_mode=args.copy_mode,
            overwrite=args.overwrite,
            copy_depths=args.copy_depths,
        )
        print(
            f"Wrote {count} mapped images and {point_count} filtered points "
            f"as COLMAP {ext} model to {Path(args.output).resolve()}"
        )
    else:
        count, ext = build_output(
            src_root=Path(args.input).resolve(),
            dst_root=Path(args.output).resolve(),
            method=args.method,
            dims=args.dims,
            copy_mode=args.copy_mode,
            overwrite=args.overwrite,
            copy_depths=args.copy_depths,
        )
        print(f"Wrote {count} ordered images as COLMAP {ext} model to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
