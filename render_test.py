#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import torch
import torchvision
from tqdm import tqdm
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene.colmap_loader import (
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from scene.dataset_readers import readColmapCameras
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def _has_colmap_files(folder):
    return (
        os.path.exists(os.path.join(folder, "images.bin"))
        and os.path.exists(os.path.join(folder, "cameras.bin"))
    ) or (
        os.path.exists(os.path.join(folder, "images.txt"))
        and os.path.exists(os.path.join(folder, "cameras.txt"))
    )


def _resolve_test_paths(test_path, test_images):
    test_path = os.path.abspath(test_path)

    candidates = [
        (test_path, os.path.join(test_path, "sparse", "0")),
        (test_path, os.path.join(test_path, "sparse")),
    ]

    if _has_colmap_files(test_path):
        candidates.append((os.path.dirname(test_path), test_path))

    for dataset_root, sparse_path in candidates:
        if _has_colmap_files(sparse_path):
            images_path = os.path.join(dataset_root, test_images)
            if not os.path.isdir(images_path):
                raise FileNotFoundError(
                    f"Found COLMAP files in '{sparse_path}', but image directory "
                    f"'{images_path}' does not exist. You can override it with --test_images."
                )
            return dataset_root, sparse_path

    raise FileNotFoundError(
        "Could not find a COLMAP sparse folder. Expected one of:\n"
        f"  {os.path.join(test_path, 'sparse', '0')}\n"
        f"  {os.path.join(test_path, 'sparse')}\n"
        f"  {test_path}"
    )


def _read_colmap_views(dataset_root, sparse_path, test_images, dataset_args):
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(sparse_path, "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(sparse_path, "cameras.bin"))
    except:
        cam_extrinsics = read_extrinsics_text(os.path.join(sparse_path, "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(sparse_path, "cameras.txt"))

    cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
    cam_infos = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        depths_params=None,
        images_folder=os.path.join(dataset_root, test_images),
        depths_folder="",
        test_cam_names_list=cam_names,
    )
    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
    return cameraList_from_camInfos(cam_infos, 1.0, dataset_args, False, True)


def _load_gaussians(dataset_args, iteration):
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(dataset_args.model_path, "point_cloud"))

    ply_path = os.path.join(
        dataset_args.model_path,
        "point_cloud",
        f"iteration_{iteration}",
        "point_cloud.ply",
    )
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Could not find trained model at '{ply_path}'.")

    gaussians = GaussianModel(dataset_args.sh_degree)
    gaussians.load_ply(ply_path, dataset_args.train_test_exp)
    return gaussians, iteration


def _should_use_trained_exposure(dataset_args, gaussians, views):
    if not dataset_args.train_test_exp:
        return False

    pretrained_exposures = getattr(gaussians, "pretrained_exposures", None)
    if pretrained_exposures is None:
        print("Train/test exposure was enabled, but no pretrained exposures were found. Rendering without exposure correction.")
        return False

    missing = [view.image_name for view in views if view.image_name not in pretrained_exposures]
    if missing:
        print(
            "Train/test exposure was enabled, but some test images are missing exposure parameters. "
            "Rendering without exposure correction."
        )
        print("First missing image:", missing[0])
        return False

    return True


def render_set(output_path, views, gaussians, pipeline, background, use_trained_exp):
    render_path = os.path.join(output_path, "renders")
    gt_path = os.path.join(output_path, "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    image_list_path = os.path.join(output_path, "image_list.txt")
    with open(image_list_path, "w") as image_list_file:
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(
                view,
                gaussians,
                pipeline,
                background,
                use_trained_exp=use_trained_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )["render"]
            gt = view.original_image[0:3, :, :]

            if use_trained_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]
                gt = gt[..., gt.shape[-1] // 2:]

            image_name = f"{idx:05d}.png"
            torchvision.utils.save_image(rendering, os.path.join(render_path, image_name))
            torchvision.utils.save_image(gt, os.path.join(gt_path, image_name))
            image_list_file.write(f"{image_name} {view.image_name}\n")


def render_test_set(dataset_args, test_path, test_images, iteration, pipeline, output_path):
    dataset_root, sparse_path = _resolve_test_paths(test_path, test_images)
    views = _read_colmap_views(dataset_root, sparse_path, test_images, dataset_args)
    gaussians, loaded_iteration = _load_gaussians(dataset_args, iteration)

    if not output_path:
        output_path = os.path.join(dataset_args.model_path, "test", f"ours_{loaded_iteration}")

    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    use_trained_exp = _should_use_trained_exposure(dataset_args, gaussians, views)

    print(f"Rendering {len(views)} views from '{dataset_root}'")
    print(f"Using COLMAP cameras from '{sparse_path}'")
    print(f"Saving outputs to '{output_path}'")

    with torch.no_grad():
        render_set(output_path, views, gaussians, pipeline, background, use_trained_exp)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render a custom COLMAP test set with a trained Gaussian model")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--test_path", required=True, type=str, help="Path to the test COLMAP dataset root or sparse folder")
    parser.add_argument("--test_images", type=str, default="images", help="Image subdirectory inside the test dataset root")
    parser.add_argument("--output_path", type=str, default="", help="Directory to save renders and GT images")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    if not args.model_path:
        raise ValueError("--model_path/-m is required.")

    print("Rendering " + args.model_path)
    safe_state(args.quiet)

    render_test_set(
        model.extract(args),
        args.test_path,
        args.test_images,
        args.iteration,
        pipeline.extract(args),
        args.output_path,
    )
