#!/usr/bin/env python3
"""Generate overlapped COLMAP chunks from an oracle camera graph.

输入：
  1. stage1/camera_graph.py 输出的 JSON 图
  2. 完整 COLMAP 数据集根目录，要求包含 images 和 sparse/0

输出：
  chunk_0000/
    images/
    sparse/0/{cameras,images,points3D}.bin 或 .txt
    chunk_metadata.json
    mapping.txt

每个 chunk 由两类图像组成：
  core    : 这个 chunk 真正负责覆盖的图像，生成后会被标记为 covered
  overlap : 为了和其他 chunk 共享上下文而额外引用的图像，不改变 covered 状态
"""

import argparse
import csv
import errno
import json
import os
import shutil
import struct
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import (  # noqa: E402
    Image,
    Point3D,
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    write_model,
)


MODEL_FILES = {
    "cameras.bin",
    "cameras.txt",
    "images.bin",
    "images.txt",
    "points3D.bin",
    "points3D.txt",
    "points3D.ply",
}


def detect_sparse_format(sparse_dir):
    """检测 COLMAP sparse 模型格式，优先使用 bin。"""
    if (sparse_dir / "images.bin").is_file():
        return ".bin"
    if (sparse_dir / "images.txt").is_file():
        return ".txt"
    raise FileNotFoundError(f"Cannot find images.bin or images.txt in {sparse_dir}")


def read_cameras_and_images(sparse_dir, ext):
    """读取 cameras/images；chunk 规划和图片点引用都需要 images。"""
    if ext == ".bin":
        cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
        images = read_images_binary(str(sparse_dir / "images.bin"))
    elif ext == ".txt":
        cameras = read_cameras_text(str(sparse_dir / "cameras.txt"))
        images = read_images_text(str(sparse_dir / "images.txt"))
    else:
        raise ValueError(f"Unsupported COLMAP extension: {ext}")
    return cameras, images


def read_points3d_subset_binary(path, needed_point_ids):
    """从 points3D.bin 中只读取 needed_point_ids 对应的点。

    文件仍需顺序扫描一遍，因为 COLMAP binary 是变长 track 布局；但对不需要
    的点直接 seek 跳过 track，不构建 Python 对象，速度和内存都比完整读取更好。
    """
    needed_point_ids = set(needed_point_ids)
    points3D = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            props = struct.unpack("<QdddBBBd", f.read(43))
            point_id = props[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            if point_id not in needed_point_ids:
                f.seek(8 * track_length, os.SEEK_CUR)
                continue

            track = struct.unpack("<" + "ii" * track_length, f.read(8 * track_length))
            points3D[point_id] = Point3D(
                id=point_id,
                xyz=np.array(props[1:4]),
                rgb=np.array(props[4:7]),
                error=np.array(props[7]),
                image_ids=np.array(tuple(map(int, track[0::2]))),
                point2D_idxs=np.array(tuple(map(int, track[1::2]))),
            )
    return points3D


def read_points3d_subset_text(path, needed_point_ids):
    """从 points3D.txt 中只解析 needed_point_ids 对应的点。"""
    needed_point_ids = set(needed_point_ids)
    points3D = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue
            elems = line.split()
            point_id = int(elems[0])
            if point_id not in needed_point_ids:
                continue
            points3D[point_id] = Point3D(
                id=point_id,
                xyz=np.array(tuple(map(float, elems[1:4]))),
                rgb=np.array(tuple(map(int, elems[4:7]))),
                error=float(elems[7]),
                image_ids=np.array(tuple(map(int, elems[8::2]))),
                point2D_idxs=np.array(tuple(map(int, elems[9::2]))),
            )
    return points3D


def read_points3d_subset(sparse_dir, ext, needed_point_ids):
    """根据 chunk 需求读取 points3D 子集。"""
    if ext == ".bin":
        return read_points3d_subset_binary(sparse_dir / "points3D.bin", needed_point_ids)
    if ext == ".txt":
        return read_points3d_subset_text(sparse_dir / "points3D.txt", needed_point_ids)
    raise ValueError(f"Unsupported COLMAP extension: {ext}")


def load_graph(graph_path):
    """读取 oracle graph，并按 score 从高到低规范化邻居列表。"""
    with Path(graph_path).open("r", encoding="utf-8") as f:
        graph = json.load(f)

    normalized = {}
    for image_name, neighbors in graph.items():
        normalized[image_name] = sorted(
            neighbors,
            key=lambda item: (
                -float(item.get("score", 0.0)),
                float(item.get("dist", 0.0)),
                item.get("image", ""),
            ),
        )
    return normalized


def copy_file(src, dst, mode):
    """按指定方式把图像放入 chunk/images。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError as exc:
            # hardlink 不能跨文件系统；这种情况下退回 copy，保证输出仍可用。
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")


def copy_sparse_extra_files(src_sparse, dst_sparse):
    """复制 sparse/0 里非核心模型文件，例如自定义配置文件。"""
    dst_sparse.mkdir(parents=True, exist_ok=True)
    for src in src_sparse.iterdir():
        if src.name in MODEL_FILES:
            continue
        dst = dst_sparse / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def choose_seed(uncovered, graph):
    """从未覆盖图像中确定性选择一个 seed。

    这里用“图中边数量多的优先，名字作为 tie-breaker”，这样 seed 更倾向于
    位于局部连通区域内部，chunk 扩展时更容易达到目标大小。
    """
    return min(uncovered, key=lambda name: (-len(graph.get(name, [])), name))


def expand_core(seed, uncovered, graph, chunk_size):
    """从 seed 出发，按图边权贪心扩展 core，直到达到 chunk_size 或无候选。"""
    core = [seed]
    core_set = {seed}
    uncovered_set = set(uncovered)

    while len(core) < chunk_size:
        best_name = None
        best_key = None

        # 当前 core 中任意图像的高分邻居都可以成为下一个 core 图像。
        # 如果同一候选被多条边连接，只保留它的最高边权作为排序依据。
        for src_name in core:
            for edge in graph.get(src_name, []):
                dst_name = edge["image"]
                if dst_name in core_set or dst_name not in uncovered_set:
                    continue
                key = (
                    float(edge.get("score", 0.0)),
                    -float(edge.get("dist", 0.0)),
                    dst_name,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_name = dst_name

        if best_name is None:
            break
        core.append(best_name)
        core_set.add(best_name)

    return core


def collect_overlap(core, graph, covered, image_names, min_overlap, overlap_ratio):
    """为 core 选择 overlap 图像。

    简单策略：
      1. 优先选择已经 covered 的高分邻居，让相邻 chunk 之间共享图像；
      2. 如果数量不足，再从未在 core 内的高分邻居补足。
    """
    target_overlap = max(min_overlap, int(round(len(core) * overlap_ratio)))
    if target_overlap <= 0:
        return []

    core_set = set(core)
    valid_names = set(image_names)
    scored = {}

    def add_candidate(edge, priority):
        name = edge["image"]
        if name in core_set or name not in valid_names:
            return
        score = float(edge.get("score", 0.0))
        dist = float(edge.get("dist", 0.0))
        key = (priority, score, -dist, name)
        if name not in scored or key > scored[name]:
            scored[name] = key

    for src_name in core:
        for edge in graph.get(src_name, []):
            # 已覆盖图像优先作为 overlap；新图像也允许补足最小 overlap。
            priority = 1 if edge["image"] in covered else 0
            add_candidate(edge, priority)

    ordered = sorted(scored, key=lambda name: scored[name], reverse=True)
    return ordered[:target_overlap]


def make_chunks(graph, image_names, chunk_size, min_overlap, overlap_ratio, max_chunks):
    """生成 chunk 规划，不触碰磁盘。"""
    graph_names = set(graph.keys())
    image_name_set = set(image_names)
    missing_in_colmap = sorted(graph_names - image_name_set)
    if missing_in_colmap:
        raise KeyError(
            f"{len(missing_in_colmap)} graph images are not in COLMAP, "
            f"first missing: {missing_in_colmap[0]}"
        )

    uncovered = set(image_names)
    covered = set()
    chunks = []

    while uncovered and (max_chunks is None or len(chunks) < max_chunks):
        seed = choose_seed(uncovered, graph)
        core = expand_core(seed, uncovered, graph, chunk_size)
        if not core:
            core = [seed]

        overlap = collect_overlap(
            core=core,
            graph=graph,
            covered=covered,
            image_names=image_names,
            min_overlap=min_overlap,
            overlap_ratio=overlap_ratio,
        )

        chunk = {
            "name": f"chunk_{len(chunks):04d}",
            "seed": seed,
            "core": core,
            "overlap": overlap,
            "images": core + [name for name in overlap if name not in set(core)],
        }
        chunks.append(chunk)

        covered.update(core)
        uncovered.difference_update(core)

    return chunks


def filter_colmap_model(cameras, images, points3D, selected_names):
    """从完整 COLMAP 模型中过滤出一个合法的子模型。

    关键点：
      - 只保留 selected_names 对应的 images；
      - cameras 只保留这些 images 使用到的 camera_id；
      - points3D 只保留被 selected images 观测到的点；
      - 每个 Point3D 的 track 只保留 selected images 内的观测；
      - image.point3D_ids 中不再存在的点改成 -1，避免悬空引用。
    """
    selected_names = set(selected_names)
    selected_images = OrderedDict(
        (image_id, image)
        for image_id, image in images.items()
        if image.name in selected_names
    )
    if len(selected_images) != len(selected_names):
        found = {image.name for image in selected_images.values()}
        missing = sorted(selected_names - found)
        raise KeyError(f"Selected image is not in COLMAP model: {missing[0]}")

    selected_ids = list(selected_images.keys())
    selected_id_array = np.array(selected_ids, dtype=np.int64)
    used_camera_ids = {image.camera_id for image in selected_images.values()}
    filtered_cameras = OrderedDict(
        (camera_id, camera)
        for camera_id, camera in cameras.items()
        if camera_id in used_camera_ids
    )

    referenced_point_ids = set()
    for image in selected_images.values():
        referenced_point_ids.update(int(pid) for pid in image.point3D_ids if int(pid) >= 0)

    filtered_points = OrderedDict()
    for point_id in sorted(referenced_point_ids):
        point = points3D.get(point_id)
        if point is None:
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
    filtered_images = OrderedDict()
    for image_id, image in selected_images.items():
        point3D_ids = np.array(
            [int(pid) if int(pid) in kept_point_ids else -1 for pid in image.point3D_ids],
            dtype=image.point3D_ids.dtype,
        )
        filtered_images[image_id] = Image(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            xys=image.xys,
            point3D_ids=point3D_ids,
        )

    return filtered_cameras, filtered_images, filtered_points


def write_chunk_mapping(chunk_root, src_images_dir, chunk):
    """记录 chunk 内图片对应的原始数据集图片。

    当前脚本不会重命名图片，因此 chunk_image_name 和 original_image_name
    通常相同；仍然显式写出两列，方便以后如果加入重命名逻辑也保持格式稳定。
    """
    core_set = set(chunk["core"])
    overlap_set = set(chunk["overlap"])
    mapping_path = chunk_root / "mapping.txt"
    with mapping_path.open("w", encoding="utf-8") as f:
        f.write("# chunk_image_name original_image_name original_image_path role\n")
        for image_name in chunk["images"]:
            if image_name in core_set:
                role = "core"
            elif image_name in overlap_set:
                role = "overlap"
            else:
                role = "unknown"
            original_path = (src_images_dir / image_name).resolve()
            f.write(f"{image_name} {image_name} {original_path} {role}\n")


def write_chunk(
    chunk,
    src_root,
    dst_root,
    cameras,
    images,
    points3D,
    ext,
    copy_mode,
):
    """写一个 chunk 文件夹。"""
    src_images_dir = src_root / "images"
    src_sparse_dir = src_root / "sparse" / "0"
    chunk_root = dst_root / chunk["name"]
    dst_images_dir = chunk_root / "images"
    dst_sparse_dir = chunk_root / "sparse" / "0"

    dst_images_dir.mkdir(parents=True, exist_ok=True)
    copy_sparse_extra_files(src_sparse_dir, dst_sparse_dir)

    selected_names = chunk["images"]
    selected_cameras, selected_images, selected_points = filter_colmap_model(
        cameras, images, points3D, selected_names
    )

    for image_name in selected_names:
        src = src_images_dir / image_name
        dst = dst_images_dir / image_name
        if not src.is_file():
            raise FileNotFoundError(f"Image referenced by COLMAP is missing: {src}")
        copy_file(src, dst, copy_mode)

    write_model(selected_cameras, selected_images, selected_points, str(dst_sparse_dir), ext=ext)
    write_chunk_mapping(chunk_root, src_images_dir, chunk)

    metadata = {
        "chunk": chunk["name"],
        "seed": chunk["seed"],
        "num_core": len(chunk["core"]),
        "num_overlap": len(chunk["overlap"]),
        "num_images": len(chunk["images"]),
        "num_points3D": len(selected_points),
        "core": chunk["core"],
        "overlap": chunk["overlap"],
    }
    with (chunk_root / "chunk_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return metadata


def write_summary(output_root, metadata_rows):
    """写总览 CSV，方便快速检查每个 chunk 的规模。"""
    with (output_root / "chunks_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk", "seed", "num_core", "num_overlap", "num_images", "num_points3D"])
        for item in metadata_rows:
            writer.writerow(
                [
                    item["chunk"],
                    item["seed"],
                    item["num_core"],
                    item["num_overlap"],
                    item["num_images"],
                    item["num_points3D"],
                ]
            )


def generate_chunks(
    graph_path,
    input_root,
    output_root,
    chunk_size,
    min_overlap,
    overlap_ratio,
    max_chunks,
    copy_mode,
    overwrite,
):
    """完整 pipeline：读图、规划 chunk、过滤并写出 COLMAP 子模型。"""
    input_root = Path(input_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    src_sparse = input_root / "sparse" / "0"
    src_images = input_root / "images"

    if not src_images.is_dir():
        raise FileNotFoundError(f"Cannot find images folder: {src_images}")
    if not src_sparse.is_dir():
        raise FileNotFoundError(f"Cannot find sparse/0 folder: {src_sparse}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ext = detect_sparse_format(src_sparse)
    graph = load_graph(graph_path)
    cameras, images = read_cameras_and_images(src_sparse, ext)
    image_names = [image.name for _, image in sorted(images.items())]

    chunks = make_chunks(
        graph=graph,
        image_names=image_names,
        chunk_size=chunk_size,
        min_overlap=min_overlap,
        overlap_ratio=overlap_ratio,
        max_chunks=max_chunks,
    )

    selected_names = set()
    for chunk in chunks:
        selected_names.update(chunk["images"])
    needed_point_ids = set()
    for image in images.values():
        if image.name not in selected_names:
            continue
        needed_point_ids.update(int(pid) for pid in image.point3D_ids if int(pid) >= 0)
    points3D = read_points3d_subset(src_sparse, ext, needed_point_ids)

    metadata_rows = []
    for chunk in chunks:
        metadata = write_chunk(
            chunk=chunk,
            src_root=input_root,
            dst_root=output_root,
            cameras=cameras,
            images=images,
            points3D=points3D,
            ext=ext,
            copy_mode=copy_mode,
        )
        metadata_rows.append(metadata)

    write_summary(output_root, metadata_rows)
    return output_root, ext, metadata_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate overlapped COLMAP chunks using an oracle camera graph."
    )
    parser.add_argument("--graph", required=True, help="Oracle graph JSON from stage1/camera_graph.py.")
    parser.add_argument("--input", "-i", required=True, help="Input COLMAP root containing images and sparse/0.")
    parser.add_argument("--output", "-o", required=True, help="Output folder that will contain chunk_XXXX folders.")
    parser.add_argument("--chunk_size", type=int, default=80, help="Target number of core images per chunk.")
    parser.add_argument("--min_overlap", type=int, default=0, help="Minimum overlap images per chunk.")
    parser.add_argument("--overlap_ratio", type=float, default=0.15, help="Overlap count = round(core_size * ratio), combined with min_overlap.")
    parser.add_argument("--max_chunks", type=int, default=None, help="Optional maximum number of chunks to generate.")
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to place images in each chunk/images folder.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove output folder first if it exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be > 0.")
    if args.min_overlap < 0:
        raise ValueError("--min_overlap must be >= 0.")
    if args.overlap_ratio < 0.0:
        raise ValueError("--overlap_ratio must be >= 0.")
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError("--max_chunks must be > 0 when set.")

    output_root, ext, metadata_rows = generate_chunks(
        graph_path=Path(args.graph),
        input_root=Path(args.input),
        output_root=Path(args.output),
        chunk_size=args.chunk_size,
        min_overlap=args.min_overlap,
        overlap_ratio=args.overlap_ratio,
        max_chunks=args.max_chunks,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
    )

    total_core = sum(item["num_core"] for item in metadata_rows)
    total_overlap_refs = sum(item["num_overlap"] for item in metadata_rows)
    print(f"Wrote {len(metadata_rows)} chunks to {output_root}")
    print(f"COLMAP format: {ext}")
    print(f"Core image assignments: {total_core}")
    print(f"Overlap image references: {total_overlap_refs}")


if __name__ == "__main__":
    main()
