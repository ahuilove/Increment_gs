#!/usr/bin/env python3
"""Run the MASt3R-SfM main pipeline on one image folder.

这个脚本只负责 MASt3R-SfM 主路线：
  images -> make_pairs -> sparse_global_alignment -> scene.pt

它不调用 COLMAP/GLOMAP mapper，也不导出 COLMAP sparse/0。后续可以由单独的
转换脚本读取 scene.pt，再写出 cameras/images/points3D。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MAST3R_ROOT = REPO_ROOT / "submodules" / "mast3r"
DUST3R_ROOT = MAST3R_ROOT / "dust3r"
if str(DUST3R_ROOT) not in sys.path:
    sys.path.insert(0, str(DUST3R_ROOT))
if str(MAST3R_ROOT) not in sys.path:
    sys.path.insert(0, str(MAST3R_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(images_dir):
    """按相对路径稳定排序，返回图像绝对路径列表。"""
    images_dir = Path(images_dir).expanduser().resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Cannot find images folder: {images_dir}")

    image_paths = []
    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            image_paths.append(path)
    image_paths.sort(key=lambda p: p.relative_to(images_dir).as_posix())

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {images_dir}")
    return image_paths


def build_scene_graph(scene_graph_type, winsize, win_cyclic, refid):
    """复用 MASt3R demo.py 里的 scene_graph 字符串约定。"""
    params = [scene_graph_type]
    if scene_graph_type in {"swin", "logwin"}:
        params.append(str(winsize))
    elif scene_graph_type == "oneref":
        params.append(str(refid))
    elif scene_graph_type == "retrieval":
        # retrieval-<num_key_images>-<num_neighbors>
        params.append(str(winsize))
        params.append(str(refid))

    if scene_graph_type in {"swin", "logwin"} and not win_cyclic:
        params.append("noncyclic")
    return "-".join(params)


def import_mast3r_runtime():
    """懒加载 MASt3R 依赖，让 --help 在依赖未安装时也能正常工作。"""
    try:
        from dust3r.utils.image import load_images
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        from mast3r.image_pairs import make_pairs
        from mast3r.model import AsymmetricMASt3R
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing MASt3R runtime dependency: {exc.name}. "
            "Please activate/install the MASt3R environment before running reconstruction."
        ) from exc

    try:
        from mast3r.retrieval.processor import Retriever
        has_retrieval = True
    except Exception:
        Retriever = None
        has_retrieval = False

    return {
        "load_images": load_images,
        "sparse_global_alignment": sparse_global_alignment,
        "make_pairs": make_pairs,
        "AsymmetricMASt3R": AsymmetricMASt3R,
        "Retriever": Retriever,
        "has_retrieval": has_retrieval,
    }


def load_model(weights, model_name, device, model_cls):
    """加载 MASt3R 模型；优先使用本地 weights。"""
    if weights is not None:
        weights_path = Path(weights).expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Cannot find weights: {weights_path}. "
                "Please place the checkpoint under stage1/checkpoint first."
            )
        model_source = str(weights_path)
    else:
        # 这里不会主动下载到 stage1/checkpoint；如果本地 HF 缓存没有模型，
        # from_pretrained 可能会访问网络。推荐显式传 --weights。
        model_source = "naver/" + model_name

    model = model_cls.from_pretrained(model_source).to(device)
    model.eval()
    return model, model_source


def tensor_to_cpu(value):
    """把 scene 结果转成 torch.save 友好的 CPU 数据。"""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, list):
        return [tensor_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(tensor_to_cpu(v) for v in value)
    if isinstance(value, dict):
        return {k: tensor_to_cpu(v) for k, v in value.items()}
    return value


def run_mast3r_sfm(args):
    runtime = import_mast3r_runtime()
    images_dir = Path(args.images).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(images_dir)
    filelist = [str(path) for path in image_paths]

    model, model_source = load_model(
        args.weights, args.model_name, args.device, runtime["AsymmetricMASt3R"]
    )
    imgs = runtime["load_images"](filelist, size=args.image_size, verbose=not args.silent)

    scene_graph = build_scene_graph(
        scene_graph_type=args.scene_graph,
        winsize=args.winsize,
        win_cyclic=args.win_cyclic,
        refid=args.refid,
    )

    sim_matrix = None
    if args.scene_graph == "retrieval":
        if not runtime["has_retrieval"]:
            raise RuntimeError("Retrieval scene graph requested, but retrieval dependencies are unavailable.")
        if args.retrieval_model is None:
            raise ValueError("--retrieval_model is required when --scene_graph retrieval.")
        retriever = runtime["Retriever"](args.retrieval_model, backbone=model, device=args.device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)
        del retriever
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    pairs = runtime["make_pairs"](imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)
    if len(pairs) == 0:
        raise RuntimeError(f"No image pairs were generated for scene_graph={scene_graph}")

    scene = runtime["sparse_global_alignment"](
        filelist,
        pairs,
        str(cache_dir),
        model,
        subsample=args.subsample,
        desc_conf=args.desc_conf,
        kinematic_mode=args.kinematic_mode,
        device=args.device,
        shared_intrinsics=args.shared_intrinsics,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=args.lr2,
        niter2=args.niter2,
        matching_conf_thr=args.matching_conf_thr,
        opt_depth=args.opt_depth,
        verbose=not args.silent,
    )

    scene_data = {
        "image_paths": filelist,
        "image_names": [path.relative_to(images_dir).as_posix() for path in image_paths],
        "images_dir": str(images_dir),
        "intrinsics": tensor_to_cpu(scene.intrinsics),
        "cam2world": tensor_to_cpu(scene.get_im_poses()),
        "focals": tensor_to_cpu(scene.get_focals()),
        "principal_points": tensor_to_cpu(scene.get_principal_points()),
        "sparse_pts3d": tensor_to_cpu(scene.get_sparse_pts3d()),
        "sparse_colors": tensor_to_cpu(scene.get_pts3d_colors()),
        "depthmaps": tensor_to_cpu(scene.get_depthmaps()),
        "config": {
            "model_source": model_source,
            "image_size": args.image_size,
            "scene_graph": scene_graph,
            "subsample": args.subsample,
            "desc_conf": args.desc_conf,
            "kinematic_mode": args.kinematic_mode,
            "shared_intrinsics": args.shared_intrinsics,
            "lr1": args.lr1,
            "niter1": args.niter1,
            "lr2": args.lr2,
            "niter2": args.niter2,
            "matching_conf_thr": args.matching_conf_thr,
            "opt_depth": args.opt_depth,
        },
    }

    scene_path = output_dir / "scene.pt"
    torch.save(scene_data, scene_path)

    with (output_dir / "image_list.txt").open("w", encoding="utf-8") as f:
        for name in scene_data["image_names"]:
            f.write(name + "\n")
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(scene_data["config"], f, indent=2)
        f.write("\n")

    return scene_path, len(image_paths), len(pairs)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MASt3R-SfM sparse_global_alignment on an image folder.")
    parser.add_argument("--images", required=True, help="Input image folder, e.g. stage1/chunk/chunk_0000/images.")
    parser.add_argument("--output", required=True, help="Output folder for scene.pt and MASt3R cache.")
    parser.add_argument("--cache_dir", default=None, help="Optional cache folder. Default: <output>/cache.")
    parser.add_argument("--weights", default=None, help="Local MASt3R checkpoint, recommended under stage1/checkpoint/.")
    parser.add_argument(
        "--model_name",
        default="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        choices=["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"],
        help="HuggingFace model name suffix used only when --weights is not set.",
    )
    parser.add_argument("--retrieval_model", default=None, help="Retrieval checkpoint for --scene_graph retrieval.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument("--image_size", type=int, default=512, help="MASt3R input resize size.")
    parser.add_argument(
        "--scene_graph",
        choices=["complete", "swin", "logwin", "oneref", "retrieval"],
        default="swin",
        help="Pair graph type passed to mast3r.image_pairs.make_pairs.",
    )
    parser.add_argument("--winsize", type=int, default=5, help="Window/key-image parameter for swin/logwin/retrieval.")
    parser.add_argument("--win_cyclic", action="store_true", help="Use cyclic window graph for swin/logwin.")
    parser.add_argument("--refid", type=int, default=0, help="Reference id for oneref, or neighbors for retrieval.")
    parser.add_argument("--subsample", type=int, default=8, help="Sparse correspondence subsample used by MASt3R-SfM.")
    parser.add_argument("--desc_conf", default="desc_conf", help="Descriptor confidence key.")
    parser.add_argument("--kinematic_mode", default="hclust-ward", help="MASt3R-SfM kinematic init mode.")
    parser.add_argument("--shared_intrinsics", action="store_true", help="Optimize one shared intrinsic model.")
    parser.add_argument("--lr1", type=float, default=0.07, help="Coarse optimization learning rate.")
    parser.add_argument("--niter1", type=int, default=300, help="Coarse optimization iterations.")
    parser.add_argument("--lr2", type=float, default=0.01, help="Fine optimization learning rate.")
    parser.add_argument("--niter2", type=int, default=300, help="Fine optimization iterations.")
    parser.add_argument("--matching_conf_thr", type=float, default=0.0, help="MASt3R matching confidence threshold.")
    parser.add_argument("--opt_depth", action="store_true", help="Optimize depth during fine stage.")
    parser.add_argument("--silent", action="store_true", help="Reduce logging.")
    return parser.parse_args()


def main():
    args = parse_args()
    scene_path, num_images, num_pairs = run_mast3r_sfm(args)
    print(f"Loaded {num_images} images")
    print(f"Generated {num_pairs} image pairs")
    print(f"Wrote MASt3R-SfM scene to {scene_path}")


if __name__ == "__main__":
    main()
