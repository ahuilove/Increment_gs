#!/usr/bin/env python3
"""Export MASt3R-SfM scene.pt to simple PLY files for offline visualization.

输出：
  mast3r_sparse_points.ply   colored sparse points from MASt3R-SfM
  mast3r_cameras.ply         camera centers and frustum wireframes
  mast3r_scene_summary.json  basic scene statistics
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


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


def normalize_colors(colors, count):
    if colors is None:
        return np.full((count, 3), 255, dtype=np.uint8)
    colors = np.asarray(colors)
    if colors.dtype.kind == "f":
        if colors.max(initial=0.0) <= 1.0:
            colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    if len(colors) != count:
        colors = np.resize(colors, (count, 3)).astype(np.uint8)
    return colors


def sample_indices(count, max_count):
    """稳定均匀抽样，避免单张图点太多导致 PLY 过大。"""
    if max_count is None or count <= max_count:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, max_count).round().astype(np.int64)


def write_point_ply(path, points, colors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for xyz, rgb in zip(points, colors):
            f.write(
                f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )


def write_line_ply(path, vertices, colors, edges):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    edges = np.asarray(edges, dtype=np.int64)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")
        for xyz, rgb in zip(vertices, colors):
            f.write(
                f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )
        for edge in edges:
            f.write(f"{int(edge[0])} {int(edge[1])}\n")


def camera_color(index, total):
    """生成稳定的相机颜色，避免所有相机线框混在一起。"""
    if total <= 1:
        t = 0.0
    else:
        t = index / (total - 1)
    r = int(255 * t)
    g = int(180 * (1.0 - abs(t - 0.5) * 2.0))
    b = int(255 * (1.0 - t))
    return np.array([r, g, b], dtype=np.uint8)


def build_camera_frustums(cam2world, intrinsics, cam_size):
    """生成相机中心和简单视锥线框。

    线框在相机坐标系中使用 +Z 作为朝前方向，再通过 cam2world 变换到世界坐标。
    """
    vertices = []
    colors = []
    edges = []
    total = len(cam2world)

    for idx, pose in enumerate(cam2world):
        pose = np.asarray(pose, dtype=np.float64)
        K = np.asarray(intrinsics[idx], dtype=np.float64)
        color = camera_color(idx, total)

        fx = max(float(K[0, 0]), 1e-6)
        fy = max(float(K[1, 1]), 1e-6)
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        # 用主点估计一个归一化成像平面大小；只用于可视化，不影响重建。
        half_w = max(cx / fx, 0.5) * cam_size
        half_h = max(cy / fy, 0.5) * cam_size
        z = cam_size

        local = np.array(
            [
                [0.0, 0.0, 0.0],
                [-half_w, -half_h, z],
                [half_w, -half_h, z],
                [half_w, half_h, z],
                [-half_w, half_h, z],
            ],
            dtype=np.float64,
        )
        homog = np.concatenate([local, np.ones((len(local), 1), dtype=np.float64)], axis=1)
        world = (pose @ homog.T).T[:, :3]

        base = len(vertices)
        vertices.extend(world.tolist())
        colors.extend([color] * len(world))
        edges.extend(
            [
                [base + 0, base + 1],
                [base + 0, base + 2],
                [base + 0, base + 3],
                [base + 0, base + 4],
                [base + 1, base + 2],
                [base + 2, base + 3],
                [base + 3, base + 4],
                [base + 4, base + 1],
            ]
        )

    return np.asarray(vertices), np.asarray(colors), np.asarray(edges)


def export_scene(scene, output_dir, max_points_per_image, cam_size):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse_pts3d = scene["sparse_pts3d"]
    sparse_colors = scene.get("sparse_colors", [None] * len(sparse_pts3d))
    all_points = []
    all_colors = []
    per_image_counts = []

    for pts_value, colors_value in zip(sparse_pts3d, sparse_colors):
        pts = to_numpy(pts_value, dtype=np.float64)
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        colors = normalize_colors(colors_value, len(valid))[valid]

        keep = sample_indices(len(pts), max_points_per_image)
        pts = pts[keep]
        colors = colors[keep]
        per_image_counts.append(int(len(pts)))

        if len(pts) > 0:
            all_points.append(pts)
            all_colors.append(colors)

    if all_points:
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
    else:
        points = np.empty((0, 3), dtype=np.float64)
        colors = np.empty((0, 3), dtype=np.uint8)

    point_ply = output_dir / "mast3r_sparse_points.ply"
    write_point_ply(point_ply, points, colors)

    cam2world = to_numpy(scene["cam2world"], dtype=np.float64)
    intrinsics = to_numpy(scene["intrinsics"], dtype=np.float64)
    cam_vertices, cam_colors, cam_edges = build_camera_frustums(cam2world, intrinsics, cam_size)
    camera_ply = output_dir / "mast3r_cameras.ply"
    write_line_ply(camera_ply, cam_vertices, cam_colors, cam_edges)

    summary = {
        "source_images": scene.get("image_names", []),
        "num_images": len(scene.get("image_names", [])),
        "num_sparse_points_exported": int(len(points)),
        "max_points_per_image": max_points_per_image,
        "per_image_exported_counts": per_image_counts,
        "camera_vertices": int(len(cam_vertices)),
        "camera_edges": int(len(cam_edges)),
        "files": {
            "points": str(point_ply),
            "cameras": str(camera_ply),
        },
    }
    with (output_dir / "mast3r_scene_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Export MASt3R-SfM scene.pt to PLY for local visualization.")
    parser.add_argument("--scene", required=True, help="Input scene.pt from mast3r_sfm_reconstruct.py.")
    parser.add_argument("--output", required=True, help="Output folder for PLY files.")
    parser.add_argument(
        "--max_points_per_image",
        type=int,
        default=5000,
        help="Maximum sparse points exported from each image. Use <=0 to export all points.",
    )
    parser.add_argument(
        "--cam_size",
        type=float,
        default=0.2,
        help="Camera frustum size in MASt3R scene units.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    max_points = None if args.max_points_per_image <= 0 else args.max_points_per_image
    scene = load_scene(args.scene)
    summary = export_scene(scene, args.output, max_points, args.cam_size)
    print(f"Wrote {summary['num_sparse_points_exported']} sparse points")
    print(f"Wrote point PLY: {summary['files']['points']}")
    print(f"Wrote camera PLY: {summary['files']['cameras']}")


if __name__ == "__main__":
    main()
