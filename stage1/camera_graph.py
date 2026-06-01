#!/usr/bin/env python3
"""Build an oracle camera graph from a COLMAP sparse model.

每个图像是一个节点；每个节点只保留空间距离和视角朝向最相关的 topK
邻居。输出的 JSON 可以直接作为 overlap chunk 的索引：以某张图像为中心，
它的邻居列表就是这个中心图像对应的重叠局部 chunk。
"""

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np


# 允许直接运行 `python stage1/camera_graph.py ...` 时也能导入项目内工具。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import (  # noqa: E402
    read_cameras_binary,
    read_cameras_text,
    qvec2rotmat,
)


def resolve_sparse_dir(input_path, sparse_subdir="sparse/0"):
    """兼容输入数据集根目录或直接输入 sparse/0 目录两种形式。"""
    input_path = Path(input_path).expanduser().resolve()

    # 如果用户直接给了 sparse/0，并且里面有 COLMAP 文件，就直接使用。
    if (input_path / "images.bin").is_file() or (input_path / "images.txt").is_file():
        return input_path

    sparse_dir = input_path / sparse_subdir
    if (sparse_dir / "images.bin").is_file() or (sparse_dir / "images.txt").is_file():
        return sparse_dir

    raise FileNotFoundError(
        "Cannot find COLMAP sparse model. Expected images.bin/images.txt in "
        f"{input_path} or {sparse_dir}."
    )


def read_image_poses_binary(path):
    """轻量读取 COLMAP images.bin，只保留 image 位姿，不解析 2D 点观测。"""
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            props = struct.unpack("<idddddddi", f.read(64))
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]

            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8")

            # 每个 2D 点是 (double x, double y, int64 point3D_id)，共 24 字节。
            # 构图只需要相机位姿，因此直接 seek 跳过这部分，避免大场景 IO/解包开销。
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * num_points2d, os.SEEK_CUR)

            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
    return images


def read_image_poses_text(path):
    """轻量读取 COLMAP images.txt，只保留第一行的 image 位姿。"""
    images = {}
    with open(path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue

            elems = line.split()
            image_id = int(elems[0])
            images[image_id] = {
                "qvec": np.array(tuple(map(float, elems[1:5])), dtype=np.float64),
                "tvec": np.array(tuple(map(float, elems[5:8])), dtype=np.float64),
                "camera_id": int(elems[8]),
                "name": elems[9],
            }
            f.readline()  # 第二行是 POINTS2D，当前功能不需要。
    return images


def camera_center_and_forward(qvec, tvec, image_name):
    """从 COLMAP 外参计算相机中心和世界坐标系下的 forward 方向。

    COLMAP 的 images 文件存的是 world-to-camera 位姿：
        x_cam = R * x_world + t

    因此相机中心为：
        C = -R.T * t

    COLMAP/OpenCV 相机坐标系中 +Z 是成像前方，所以世界坐标系下的朝向为：
        forward_world = R.T * [0, 0, 1]
    """
    rotation = qvec2rotmat(qvec)
    center = -rotation.T @ tvec
    forward = rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward_norm = np.linalg.norm(forward)
    if forward_norm == 0.0:
        raise ValueError(f"Image {image_name} has an invalid zero forward vector.")
    return center.astype(np.float64), forward / forward_norm


def load_camera_poses(sparse_dir):
    """读取 COLMAP sparse 模型，并整理成按 image_id 稳定排序的相机位姿数组。"""
    # 这里只需要 cameras 和 images；不要调用 read_model，因为它还会读取
    # points3D.bin。大场景的点云可能非常大，会让构图前的 IO 变慢很多。
    if (sparse_dir / "cameras.bin").is_file() and (sparse_dir / "images.bin").is_file():
        cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
        images = read_image_poses_binary(str(sparse_dir / "images.bin"))
    elif (sparse_dir / "cameras.txt").is_file() and (sparse_dir / "images.txt").is_file():
        cameras = read_cameras_text(str(sparse_dir / "cameras.txt"))
        images = read_image_poses_text(str(sparse_dir / "images.txt"))
    else:
        raise FileNotFoundError(
            f"Cannot find cameras/images COLMAP files in {sparse_dir}."
        )

    if not images:
        raise ValueError(f"No registered images found in {sparse_dir}.")

    names = []
    centers = []
    forwards = []

    for image_id in sorted(images):
        image = images[image_id]
        if image["camera_id"] not in cameras:
            raise ValueError(
                f"Image {image['name']} references missing camera_id {image['camera_id']}."
            )

        center, forward = camera_center_and_forward(
            image["qvec"], image["tvec"], image["name"]
        )
        names.append(image["name"])
        centers.append(center)
        forwards.append(forward)

    return names, np.stack(centers, axis=0), np.stack(forwards, axis=0)


def build_camera_graph(
    names,
    centers,
    forwards,
    topk,
    max_dist,
    max_angle_deg,
    sigma_d,
):
    """根据距离和朝向为每张图像选择 topK 邻居。"""
    if topk < 0:
        raise ValueError("--topk must be >= 0.")
    if sigma_d <= 0.0:
        raise ValueError("--sigma_d must be > 0.")
    if max_dist <= 0.0:
        raise ValueError("--max_dist must be > 0.")
    if not (0.0 <= max_angle_deg <= 180.0):
        raise ValueError("--max_angle_deg must be in [0, 180].")

    graph = {}
    cos_min = math.cos(math.radians(max_angle_deg))

    for i, name in enumerate(names):
        # 一次性计算当前相机到所有相机的距离和朝向夹角。
        delta = centers - centers[i]
        dists = np.linalg.norm(delta, axis=1)
        dots = forwards @ forwards[i]
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))

        # 排除自身，并应用用户给定的空间距离/视角夹角阈值。
        valid = np.ones(len(names), dtype=bool)
        valid[i] = False
        valid &= dists <= max_dist
        valid &= dots >= cos_min

        candidate_indices = np.nonzero(valid)[0]
        if topk == 0 or len(candidate_indices) == 0:
            graph[name] = []
            continue

        # 基础打分：
        # score(i, j) = exp(-dist(i,j) / sigma_d) * max(0, dot(forward_i, forward_j))
        # 先用 numpy 对所有候选边批量算分，再只取 topK，避免为全部候选创建 dict 并排序。
        candidate_scores = np.exp(-dists[candidate_indices] / sigma_d) * np.maximum(
            0.0, dots[candidate_indices]
        )
        if len(candidate_indices) > topk:
            top_local_indices = np.argpartition(-candidate_scores, topk - 1)[:topk]
            selected_indices = candidate_indices[top_local_indices]
        else:
            selected_indices = candidate_indices

        # 分数越高越相关；分数相同则优先距离更近，再按图像名稳定排序。
        neighbors = []
        for j in selected_indices:
            score = math.exp(-float(dists[j]) / sigma_d) * max(0.0, float(dots[j]))
            neighbors.append(
                {
                    "image": names[j],
                    "score": score,
                    "dist": float(dists[j]),
                    "angle_deg": float(angles[j]),
                }
            )
        neighbors.sort(key=lambda item: (-item["score"], item["dist"], item["image"]))
        graph[name] = [
            {
                "image": item["image"],
                "score": round(item["score"], 6),
                "dist": round(item["dist"], 6),
                "angle_deg": round(item["angle_deg"], 6),
            }
            for item in neighbors
        ]

    return graph


def write_graph(graph, output_path):
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a topK oracle camera graph from COLMAP sparse/0."
    )
    parser.add_argument(
        "input",
        help="COLMAP dataset root, e.g. /path/to/block_all, or the sparse/0 directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path. Default: <input>/oracle_camera_graph.json.",
    )
    parser.add_argument(
        "--sparse_subdir",
        default="sparse/0",
        help="Sparse model sub-directory when input is a dataset root. Default: sparse/0.",
    )
    parser.add_argument("--topk", type=int, default=20, help="Keep topK edges per image.")
    parser.add_argument(
        "--max_dist",
        type=float,
        default=float("inf"),
        help="Maximum camera-center distance for an edge. Default: no distance limit.",
    )
    parser.add_argument(
        "--max_angle_deg",
        type=float,
        default=90.0,
        help="Maximum forward-direction angle in degrees. Default: 90.",
    )
    parser.add_argument(
        "--sigma_d",
        type=float,
        default=10.0,
        help="Distance falloff in exp(-dist / sigma_d). Default: 10.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    sparse_dir = resolve_sparse_dir(input_path, args.sparse_subdir)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else input_path / "oracle_camera_graph.json"
    )

    names, centers, forwards = load_camera_poses(sparse_dir)
    graph = build_camera_graph(
        names=names,
        centers=centers,
        forwards=forwards,
        topk=args.topk,
        max_dist=args.max_dist,
        max_angle_deg=args.max_angle_deg,
        sigma_d=args.sigma_d,
    )
    output_path = write_graph(graph, output_path)

    edge_count = sum(len(neighbors) for neighbors in graph.values())
    print(f"Loaded {len(names)} images from {sparse_dir}")
    print(f"Wrote {edge_count} directed edges to {output_path}")


if __name__ == "__main__":
    main()
