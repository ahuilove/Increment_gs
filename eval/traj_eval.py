from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.utils.traj_utils import align_trajectory, compute_ate, compute_rpe, plot_trajectories, save_json


def load_read_write_model():
    """Load the project COLMAP reader without being shadowed by eval/utils."""
    module_path = REPO_ROOT / "utils" / "read_write_model.py"
    spec = importlib.util.spec_from_file_location("project_read_write_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load COLMAP reader from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_read_write_model = load_read_write_model()
qvec2rotmat = _read_write_model.qvec2rotmat
read_model = _read_write_model.read_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a predicted COLMAP camera trajectory against a ground-truth COLMAP trajectory."
    )
    parser.add_argument(
        "--pred-sparse-dir",
        required=True,
        help="Predicted COLMAP sparse model path. Can be sparse/0, sparse, or a dataset root containing sparse/0.",
    )
    parser.add_argument(
        "--gt-sparse-dir",
        required=True,
        help="Ground-truth COLMAP sparse model path. Can be sparse/0, sparse, or a dataset root containing sparse/0.",
    )
    parser.add_argument(
        "--output",
        default="eval/traj_eval_output",
        help="Output directory for metrics.json, metrics.txt, matched_poses.csv, and pose_vis.png.",
    )
    parser.add_argument(
        "--alignment",
        choices=["sim3", "se3", "none"],
        default="sim3",
        help="Alignment before metric computation. sim3 handles monocular scale ambiguity.",
    )
    parser.add_argument(
        "--match-by",
        choices=["basename", "name"],
        default="basename",
        help="Image matching key. basename ignores directory prefixes in COLMAP image names.",
    )
    parser.add_argument(
        "--rpe-percent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also report RPE_t multiplied by 100, matching InstantSplat's printed convention.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip writing pose_vis.png.")
    parser.add_argument(
        "--plot-video-frames",
        action="store_true",
        help="Also write incremental trajectory frames under output/pose_vid, similar to InstantSplat's vid mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_sparse = find_sparse_model(Path(args.pred_sparse_dir))
    gt_sparse = find_sparse_model(Path(args.gt_sparse_dir))

    pred_poses = load_colmap_c2w_poses(pred_sparse, match_by=args.match_by)
    gt_poses = load_colmap_c2w_poses(gt_sparse, match_by=args.match_by)
    names = sorted(set(pred_poses) & set(gt_poses))
    if len(names) < 2:
        raise ValueError(
            f"Need at least 2 matched images, found {len(names)}. "
            f"Pred images={len(pred_poses)}, GT images={len(gt_poses)}"
        )

    pred_c2w = np.stack([pred_poses[name] for name in names])
    gt_c2w = np.stack([gt_poses[name] for name in names])

    pred_aligned, alignment_info = align_trajectory(pred_c2w, gt_c2w, method=args.alignment)
    ate = compute_ate(gt_c2w, pred_aligned)
    rpe_t, rpe_r_rad = compute_rpe(gt_c2w, pred_aligned)
    rpe_r_deg = float(np.degrees(rpe_r_rad))

    metrics = {
        "pred_sparse_dir": str(pred_sparse),
        "gt_sparse_dir": str(gt_sparse),
        "num_pred_images": len(pred_poses),
        "num_gt_images": len(gt_poses),
        "num_matched_images": len(names),
        "alignment": alignment_info,
        "RPE_t": float(rpe_t),
        "RPE_t_x100": float(rpe_t * 100.0),
        "RPE_r_rad": float(rpe_r_rad),
        "RPE_r_deg": rpe_r_deg,
        "ATE": float(ate),
    }

    save_json(output_dir / "metrics.json", metrics)
    write_metrics_txt(output_dir / "metrics.txt", metrics, include_percent=args.rpe_percent)
    write_matched_poses_csv(output_dir / "matched_poses.csv", names, gt_c2w, pred_c2w, pred_aligned)

    if not args.no_plot:
        plot_trajectories(gt_c2w, pred_aligned, names, output_dir / "pose_vis.png", vid=args.plot_video_frames)

    print(f"Prediction sparse model: {pred_sparse}")
    print(f"Ground-truth sparse model: {gt_sparse}")
    print(f"Matched images: {len(names)}")
    print(f"Alignment: {args.alignment}, scale={alignment_info['scale']:.8f}")
    if args.rpe_percent:
        print(f"RPE_t: {rpe_t * 100.0:.7f} (x100)")
    else:
        print(f"RPE_t: {rpe_t:.7f}")
    print(f"RPE_r: {rpe_r_deg:.7f} deg")
    print(f"ATE  : {ate:.7f}")
    print(f"Saved outputs to: {output_dir}")


def find_sparse_model(path: Path) -> Path:
    """Accept sparse/0, sparse, or dataset root and return the model directory."""
    path = path.expanduser().resolve()
    candidates = [
        path,
        path / "0",
        path / "sparse" / "0",
    ]
    for candidate in candidates:
        if has_colmap_model_files(candidate):
            return candidate

    if path.exists():
        nested = [p for p in path.rglob("*") if p.is_dir() and has_colmap_model_files(p)]
        if nested:
            return max(nested, key=lambda p: count_images_in_model(p))

    raise FileNotFoundError(f"Could not find COLMAP cameras/images files under {path}")


def has_colmap_model_files(path: Path) -> bool:
    return (
        ((path / "images.bin").exists() or (path / "images.txt").exists())
        and ((path / "cameras.bin").exists() or (path / "cameras.txt").exists())
    )


def count_images_in_model(path: Path) -> int:
    try:
        _, images, _ = read_model(str(path), ext="")
        return len(images)
    except Exception:
        return -1


def load_colmap_c2w_poses(path: Path, match_by: str) -> dict[str, np.ndarray]:
    """Read COLMAP images and return camera-to-world poses keyed by image name."""
    _, images, _ = read_model(str(path), ext="")
    if images is None or not images:
        raise FileNotFoundError(f"No COLMAP images found in {path}")

    poses: dict[str, np.ndarray] = {}
    for image in images.values():
        key = image.name if match_by == "name" else os.path.basename(image.name)
        if key in poses:
            raise ValueError(f"Duplicate image match key '{key}' in {path}; use --match-by name.")

        # COLMAP stores world-to-camera: x_cam = R_cw * x_world + t_cw.
        # Trajectory metrics are easier to reason about in camera-to-world.
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = qvec2rotmat(image.qvec)
        w2c[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
        poses[key] = np.linalg.inv(w2c)

    return poses


def write_metrics_txt(path: Path, metrics: dict, include_percent: bool = True) -> None:
    rpe_t_value = metrics["RPE_t_x100"] if include_percent else metrics["RPE_t"]
    rpe_t_suffix = " (x100)" if include_percent else ""
    lines = [
        f"Prediction sparse model: {metrics['pred_sparse_dir']}",
        f"Ground-truth sparse model: {metrics['gt_sparse_dir']}",
        f"Matched images: {metrics['num_matched_images']}",
        f"Alignment: {metrics['alignment']['method']}",
        f"Alignment scale: {metrics['alignment']['scale']:.10f}",
        f"RPE_t{rpe_t_suffix}: {rpe_t_value:.7f}",
        f"RPE_t_raw: {metrics['RPE_t']:.7f}",
        f"RPE_r_deg: {metrics['RPE_r_deg']:.7f}",
        f"ATE: {metrics['ATE']:.7f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_matched_poses_csv(
    path: Path,
    names: list[str],
    gt_c2w: np.ndarray,
    pred_c2w: np.ndarray,
    pred_aligned_c2w: np.ndarray,
) -> None:
    """Write camera centers for quick debugging and external plotting."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "gt_x",
                "gt_y",
                "gt_z",
                "pred_x",
                "pred_y",
                "pred_z",
                "pred_aligned_x",
                "pred_aligned_y",
                "pred_aligned_z",
            ]
        )
        for name, gt_pose, pred_pose, pred_aligned_pose in zip(names, gt_c2w, pred_c2w, pred_aligned_c2w):
            writer.writerow(
                [
                    name,
                    *gt_pose[:3, 3].tolist(),
                    *pred_pose[:3, 3].tolist(),
                    *pred_aligned_pose[:3, 3].tolist(),
                ]
            )


if __name__ == "__main__":
    main()
