import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

"""
食用方法：支持colmap和colmap的比较   也支持vggt-omega的prediction.npz和colmap的比较

倘若将colmap和colmap比较，直接传入两个colmap稀疏模型路径：
python stage1/eval_metric.py \
    --pred-sparse-dir stage1/chunk/chunk_0000/sparse_vggtx \
    --sparse-dir stage1/chunk/chunk_0000/sparse_gt \
    --thresholds 3 30
倘若使用vggt-omega的prediction.npz和colmap比较，传入prediction.npz路径和colmap稀疏模型路径：
python stage1/eval_metric.py \
    --predictions demo_outputs/input_images_20260531_162151_951531/predictions.npz \
    --sparse-dir stage1/chunk/chunk_0000/sparese_gt
"""



REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import qvec2rotmat, read_model


def main() -> None:
    args = parse_args()

    demo_dir = Path(args.demo_dir)
    predictions_path = Path(args.predictions) if args.predictions else demo_dir / "predictions.npz"
    images_dir = Path(args.images_dir) if args.images_dir else demo_dir / "images"
    pred_sparse_dir = find_sparse_model(Path(args.pred_sparse_dir)) if args.pred_sparse_dir else None
    sparse_dir = find_sparse_model(Path(args.sparse_dir) if args.sparse_dir else demo_dir / "sparse")

    colmap_extrinsics, colmap_intrinsics, colmap_sizes = load_colmap_model(sparse_dir)
    pred_colmap_sizes = None
    if pred_sparse_dir is not None:
        pred_extrinsics, pred_intrinsics, pred_colmap_sizes = load_colmap_model(pred_sparse_dir)
        image_names = sorted(pred_extrinsics)
        matched = match_colmap_by_name(pred_extrinsics, pred_intrinsics, colmap_extrinsics, colmap_intrinsics)
    else:
        pred_extrinsics, pred_intrinsics = load_predictions(predictions_path)
        image_names = sorted(p.name for p in images_dir.iterdir() if p.is_file())
        if len(image_names) != len(pred_extrinsics):
            raise ValueError(
                f"Prediction count ({len(pred_extrinsics)}) does not match image count "
                f"({len(image_names)}) in {images_dir}"
            )
        matched = match_npz_by_name(image_names, pred_extrinsics, pred_intrinsics, colmap_extrinsics, colmap_intrinsics)

    if len(matched["names"]) < 2:
        raise ValueError(f"Need at least 2 matched images, found {len(matched['names'])}")

    rot_err, trans_err = pairwise_pose_errors(matched["pred_ext"], matched["gt_ext"])
    aucs = {
        f"auc@{threshold:g}": auc_min_rra_rta(rot_err, trans_err, threshold) * 100.0
        for threshold in args.thresholds
    }

    print(f"Demo dir: {demo_dir}")
    if pred_sparse_dir is not None:
        print(f"Prediction sparse model: {pred_sparse_dir}")
    else:
        print(f"Predictions: {predictions_path}")
    print(f"Sparse model: {sparse_dir}")
    print(f"Matched images: {len(matched['names'])}/{len(image_names)}")
    print(f"Pairs: {len(rot_err)}")
    print()
    for threshold in args.thresholds:
        rra = np.mean(rot_err < threshold) * 100.0
        rta = np.mean(trans_err < threshold) * 100.0
        print(f"RRA@{threshold:g}: {rra:.3f}")
        print(f"RTA@{threshold:g}: {rta:.3f}")
        print(f"AUC@{threshold:g}: {aucs[f'auc@{threshold:g}']:.3f}")
        print()

    print(f"Rotation error median/mean: {np.median(rot_err):.3f} / {np.mean(rot_err):.3f} deg")
    print(f"Translation angular error median/mean: {np.median(trans_err):.3f} / {np.mean(trans_err):.3f} deg")

    if args.eval_intrinsics:
        if pred_sparse_dir is not None:
            intrinsics_report = evaluate_colmap_intrinsics(
                matched["names"],
                matched["pred_int"],
                matched["gt_int"],
                pred_colmap_sizes,
                colmap_sizes,
            )
            print()
            print("Intrinsics in COLMAP image coordinates:")
        else:
            intrinsics_report = evaluate_intrinsics(
                matched["names"],
                matched["pred_int"],
                matched["gt_int"],
                colmap_sizes,
                images_dir,
                pred_image_size_hw=args.pred_image_size,
            )
            print()
            print("Intrinsics after mapping COLMAP K to VGGT preprocessed image coordinates:")
        print()
        for key, value in intrinsics_report.items():
            print(f"{key}: {value:.6f}")

    if args.list_missing:
        missing = sorted(set(image_names) - set(matched["names"]))
        if missing:
            print()
            print("Images missing from COLMAP:")
            for name in missing:
                print(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VGGT-Omega npz predictions or COLMAP predictions against a COLMAP sparse model."
    )
    parser.add_argument(
        "--demo-dir",
        default="demo_outputs/input_images_20260531_162151_951531",
        help="Directory containing predictions.npz, images/, and sparse/ for the original VGGT-Omega mode.",
    )
    parser.add_argument("--predictions", default=None, help="Optional path to predictions.npz.")
    parser.add_argument(
        "--pred-sparse-dir",
        default=None,
        help="Optional predicted COLMAP sparse model directory. If set, --predictions is ignored.",
    )
    parser.add_argument("--images-dir", default=None, help="Optional path to the demo images directory.")
    parser.add_argument("--sparse-dir", default=None, help="Optional ground-truth COLMAP sparse model directory.")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[3.0, 30.0],
        help="AUC thresholds in degrees.",
    )
    parser.add_argument(
        "--eval-intrinsics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also report fx/fy/FoV errors after resizing COLMAP intrinsics to VGGT coordinates.",
    )
    parser.add_argument(
        "--pred-image-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Prediction image size. Defaults to inferred size from predictions.npz intrinsics/images.",
    )
    parser.add_argument("--list-missing", action="store_true", help="List images with no COLMAP pose.")
    return parser.parse_args()


def find_sparse_model(sparse_root: Path) -> Path:
    if has_colmap_model_files(sparse_root):
        return sparse_root

    candidates = [p for p in sparse_root.iterdir() if p.is_dir() and has_colmap_model_files(p)]
    if not candidates:
        raise FileNotFoundError(f"Could not find COLMAP model files under {sparse_root}")
    if len(candidates) == 1:
        return candidates[0]

    def num_images(path: Path) -> int:
        try:
            _, images, _ = read_model(str(path), ext="")
            return len(images)
        except Exception:
            return -1

    return max(candidates, key=num_images)


def has_colmap_model_files(path: Path) -> bool:
    image_files = ["images.bin", "images.txt"]
    camera_files = ["cameras.bin", "cameras.txt"]
    return any((path / name).exists() for name in image_files) and any(
        (path / name).exists() for name in camera_files
    )


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path) as data:
        if "extrinsic" not in data or "intrinsic" not in data:
            raise KeyError(f"{path} must contain 'extrinsic' and 'intrinsic'")
        extrinsics = np.asarray(data["extrinsic"], dtype=np.float64)
        intrinsics = np.asarray(data["intrinsic"], dtype=np.float64)

        if extrinsics.ndim == 4 and extrinsics.shape[0] == 1:
            extrinsics = extrinsics[0]
        if intrinsics.ndim == 4 and intrinsics.shape[0] == 1:
            intrinsics = intrinsics[0]

    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Expected extrinsic shape [N,3,4], got {extrinsics.shape}")
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsic shape [N,3,3], got {intrinsics.shape}")
    return extrinsics, intrinsics


def load_colmap_model(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, tuple[int, int]]]:
    cameras, images, _ = read_model(str(path), ext="")
    if cameras is None or images is None:
        raise FileNotFoundError(f"Could not read COLMAP model from {path}")

    extrinsics = {}
    intrinsics = {}
    sizes = {}

    for image in images.values():
        name = os.path.basename(image.name)
        cam = cameras[image.camera_id]
        extrinsics[name] = colmap_image_to_extrinsic(image)
        intrinsics[name] = camera_to_k(cam)
        sizes[name] = (int(cam.height), int(cam.width))

    return extrinsics, intrinsics, sizes


def colmap_image_to_extrinsic(image) -> np.ndarray:
    # COLMAP stores world-to-camera pose: x_cam = R * x_world + t.
    extrinsic = np.eye(4, dtype=np.float64)[:3]
    extrinsic[:3, :3] = qvec2rotmat(image.qvec)
    extrinsic[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
    return extrinsic


def camera_to_k(camera) -> np.ndarray:
    params = np.asarray(camera.params, dtype=np.float64)
    model = camera.model
    if model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model in {"RADIAL", "RADIAL_FISHEYE"}:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
    elif model == "FOV":
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"Unsupported COLMAP camera model for intrinsics evaluation: {model}")

    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def match_npz_by_name(
    image_names: list[str],
    pred_extrinsics: np.ndarray,
    pred_intrinsics: np.ndarray,
    gt_extrinsics: dict[str, np.ndarray],
    gt_intrinsics: dict[str, np.ndarray],
) -> dict[str, np.ndarray | list[str]]:
    names = []
    pred_ext = []
    pred_int = []
    gt_ext = []
    gt_int = []

    for idx, name in enumerate(image_names):
        if name not in gt_extrinsics:
            continue
        names.append(name)
        pred_ext.append(pred_extrinsics[idx])
        pred_int.append(pred_intrinsics[idx])
        gt_ext.append(gt_extrinsics[name])
        gt_int.append(gt_intrinsics[name])

    return {
        "names": names,
        "pred_ext": np.stack(pred_ext),
        "pred_int": np.stack(pred_int),
        "gt_ext": np.stack(gt_ext),
        "gt_int": np.stack(gt_int),
    }


def match_colmap_by_name(
    pred_extrinsics: dict[str, np.ndarray],
    pred_intrinsics: dict[str, np.ndarray],
    gt_extrinsics: dict[str, np.ndarray],
    gt_intrinsics: dict[str, np.ndarray],
) -> dict[str, np.ndarray | list[str]]:
    names = sorted(set(pred_extrinsics) & set(gt_extrinsics))
    if not names:
        raise ValueError("No overlapping image names between predicted COLMAP and ground-truth COLMAP models.")

    return {
        "names": names,
        "pred_ext": np.stack([pred_extrinsics[name] for name in names]),
        "pred_int": np.stack([pred_intrinsics[name] for name in names]),
        "gt_ext": np.stack([gt_extrinsics[name] for name in names]),
        "gt_int": np.stack([gt_intrinsics[name] for name in names]),
    }


def pairwise_pose_errors(pred_ext: np.ndarray, gt_ext: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_r = pred_ext[:, :3, :3]
    pred_t = pred_ext[:, :3, 3]
    gt_r = gt_ext[:, :3, :3]
    gt_t = gt_ext[:, :3, 3]

    pred_centers = camera_centers(pred_r, pred_t)
    gt_centers = camera_centers(gt_r, gt_t)

    rot_errors = []
    trans_errors = []
    for i, j in itertools.combinations(range(len(pred_ext)), 2):
        pred_rel_r = pred_r[j] @ pred_r[i].T
        gt_rel_r = gt_r[j] @ gt_r[i].T
        rot_errors.append(rotation_angle_deg(pred_rel_r @ gt_rel_r.T))

        pred_dir = pred_r[i] @ (pred_centers[j] - pred_centers[i])
        gt_dir = gt_r[i] @ (gt_centers[j] - gt_centers[i])
        trans_errors.append(vector_angle_deg(pred_dir, gt_dir))

    return np.asarray(rot_errors), np.asarray(trans_errors)


def camera_centers(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -np.einsum("nij,nj->ni", np.swapaxes(r, 1, 2), t)


def rotation_angle_deg(r: np.ndarray) -> float:
    cos = (np.trace(r) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm <= 1e-12 or b_norm <= 1e-12:
        return 180.0
    cos = float(np.dot(a, b) / (a_norm * b_norm))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def auc_min_rra_rta(rot_err: np.ndarray, trans_err: np.ndarray, threshold: float) -> float:
    if threshold <= 0:
        raise ValueError("AUC threshold must be positive")

    # Piecewise-constant empirical accuracy curve. Include all error values up to
    # the threshold as breakpoints and integrate min(RRA, RTA) exactly.
    points = np.concatenate(([0.0], rot_err[rot_err < threshold], trans_err[trans_err < threshold], [threshold]))
    points = np.unique(np.sort(points))
    if len(points) < 2:
        return 0.0

    area = 0.0
    for left, right in zip(points[:-1], points[1:]):
        if right <= left:
            continue
        tau = (left + right) * 0.5
        acc = min(np.mean(rot_err < tau), np.mean(trans_err < tau))
        area += acc * (right - left)
    return float(area / threshold)


def evaluate_intrinsics(
    names: list[str],
    pred_k: np.ndarray,
    gt_k: np.ndarray,
    colmap_sizes: dict[str, tuple[int, int]],
    images_dir: Path,
    pred_image_size_hw: list[int] | None,
) -> dict[str, float]:
    gt_mapped = []
    for name, k in zip(names, gt_k):
        target_hw = tuple(pred_image_size_hw) if pred_image_size_hw else pred_size_from_k(pred_k[len(gt_mapped)])
        gt_mapped.append(map_colmap_k_to_prediction(k, colmap_sizes[name], images_dir / name, target_hw))
    gt_mapped = np.stack(gt_mapped)

    fx_rel = np.abs(pred_k[:, 0, 0] - gt_mapped[:, 0, 0]) / np.maximum(np.abs(gt_mapped[:, 0, 0]), 1e-12)
    fy_rel = np.abs(pred_k[:, 1, 1] - gt_mapped[:, 1, 1]) / np.maximum(np.abs(gt_mapped[:, 1, 1]), 1e-12)

    pred_fov_h, pred_fov_w = fov_from_k(pred_k)
    gt_fov_h, gt_fov_w = fov_from_k(gt_mapped)
    fov_h_err = np.abs(pred_fov_h - gt_fov_h)
    fov_w_err = np.abs(pred_fov_w - gt_fov_w)

    return {
        "fx_rel_median": float(np.median(fx_rel)),
        "fx_rel_mean": float(np.mean(fx_rel)),
        "fy_rel_median": float(np.median(fy_rel)),
        "fy_rel_mean": float(np.mean(fy_rel)),
        "fov_h_err_median_deg": float(np.median(fov_h_err)),
        "fov_h_err_mean_deg": float(np.mean(fov_h_err)),
        "fov_w_err_median_deg": float(np.median(fov_w_err)),
        "fov_w_err_mean_deg": float(np.mean(fov_w_err)),
    }


def evaluate_colmap_intrinsics(
    names: list[str],
    pred_k: np.ndarray,
    gt_k: np.ndarray,
    pred_sizes: dict[str, tuple[int, int]],
    gt_sizes: dict[str, tuple[int, int]],
) -> dict[str, float]:
    """Compare predicted and GT intrinsics directly in COLMAP image coordinates."""
    fx_rel = np.abs(pred_k[:, 0, 0] - gt_k[:, 0, 0]) / np.maximum(np.abs(gt_k[:, 0, 0]), 1e-12)
    fy_rel = np.abs(pred_k[:, 1, 1] - gt_k[:, 1, 1]) / np.maximum(np.abs(gt_k[:, 1, 1]), 1e-12)
    cx_abs = np.abs(pred_k[:, 0, 2] - gt_k[:, 0, 2])
    cy_abs = np.abs(pred_k[:, 1, 2] - gt_k[:, 1, 2])

    pred_fov_h, pred_fov_w = fov_from_k_and_sizes(pred_k, [pred_sizes[name] for name in names])
    gt_fov_h, gt_fov_w = fov_from_k_and_sizes(gt_k, [gt_sizes[name] for name in names])
    fov_h_err = np.abs(pred_fov_h - gt_fov_h)
    fov_w_err = np.abs(pred_fov_w - gt_fov_w)

    return {
        "fx_rel_median": float(np.median(fx_rel)),
        "fx_rel_mean": float(np.mean(fx_rel)),
        "fy_rel_median": float(np.median(fy_rel)),
        "fy_rel_mean": float(np.mean(fy_rel)),
        "cx_abs_median_px": float(np.median(cx_abs)),
        "cx_abs_mean_px": float(np.mean(cx_abs)),
        "cy_abs_median_px": float(np.median(cy_abs)),
        "cy_abs_mean_px": float(np.mean(cy_abs)),
        "fov_h_err_median_deg": float(np.median(fov_h_err)),
        "fov_h_err_mean_deg": float(np.mean(fov_h_err)),
        "fov_w_err_median_deg": float(np.median(fov_w_err)),
        "fov_w_err_mean_deg": float(np.mean(fov_w_err)),
    }


def pred_size_from_k(k: np.ndarray) -> tuple[int, int]:
    # VGGT-Omega builds cx=W/2 and cy=H/2.
    return int(round(2.0 * k[1, 2])), int(round(2.0 * k[0, 2]))


def map_colmap_k_to_prediction(
    k: np.ndarray,
    colmap_hw: tuple[int, int],
    image_path: Path,
    target_hw: tuple[int, int],
) -> np.ndarray:
    with Image.open(image_path) as image:
        image_w, image_h = image.size

    cam_h, cam_w = colmap_hw
    if (image_h, image_w) != (cam_h, cam_w):
        scale_x = image_w / cam_w
        scale_y = image_h / cam_h
        k = k.copy()
        k[0, 0] *= scale_x
        k[1, 1] *= scale_y
        k[0, 2] *= scale_x
        k[1, 2] *= scale_y

    crop_left, crop_top, crop_w, crop_h = supported_aspect_crop(image_w, image_h)
    target_h, target_w = target_hw
    sx = target_w / crop_w
    sy = target_h / crop_h

    mapped = k.copy()
    mapped[0, 0] *= sx
    mapped[1, 1] *= sy
    mapped[0, 2] = (mapped[0, 2] - crop_left) * sx
    mapped[1, 2] = (mapped[1, 2] - crop_top) * sy
    return mapped


def supported_aspect_crop(width: int, height: int) -> tuple[int, int, int, int]:
    aspect_ratio = height / max(width, 1)
    min_aspect_ratio = 0.5
    max_aspect_ratio = 2.0

    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return left, 0, crop_width, height

    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return 0, top, width, crop_height

    return 0, 0, width, height


def fov_from_k(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = 2.0 * k[:, 1, 2]
    width = 2.0 * k[:, 0, 2]
    fov_h = np.degrees(2.0 * np.arctan((height / 2.0) / k[:, 1, 1]))
    fov_w = np.degrees(2.0 * np.arctan((width / 2.0) / k[:, 0, 0]))
    return fov_h, fov_w


def fov_from_k_and_sizes(k: np.ndarray, sizes_hw: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.asarray(sizes_hw, dtype=np.float64)
    height = sizes[:, 0]
    width = sizes[:, 1]
    fov_h = np.degrees(2.0 * np.arctan((height / 2.0) / k[:, 1, 1]))
    fov_w = np.degrees(2.0 * np.arctan((width / 2.0) / k[:, 0, 0]))
    return fov_h, fov_w


if __name__ == "__main__":
    main()
