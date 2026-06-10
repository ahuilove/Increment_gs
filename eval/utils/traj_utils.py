from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
import matplotlib
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evo.core.trajectory import PosePath3D
from evo.tools import plot


def rotation_error_rad(transform: np.ndarray) -> float:
    """Return the angle of a rotation matrix in radians."""
    cos_theta = 0.5 * (np.trace(transform[:3, :3]) - 1.0)
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def translation_error(transform: np.ndarray) -> float:
    """Return the Euclidean length of the translation part."""
    return float(np.linalg.norm(transform[:3, 3]))


def compute_rpe(gt_c2w: np.ndarray, pred_c2w: np.ndarray) -> tuple[float, float]:
    """Compute mean adjacent-frame Relative Pose Error.

    The input poses must already be expressed in the same coordinate system.
    For every adjacent pair i -> i+1, we compare the GT relative motion with
    the predicted relative motion:

        error = inv(inv(gt_i) @ gt_j) @ (inv(pred_i) @ pred_j)

    Returns:
        (RPE_t, RPE_r_rad), where RPE_t is a translation length and RPE_r_rad is
        an angle in radians.
    """
    if len(gt_c2w) != len(pred_c2w):
        raise ValueError("GT and predicted trajectories must have the same length.")
    if len(gt_c2w) < 2:
        raise ValueError("At least two poses are required to compute RPE.")

    trans_errors = []
    rot_errors = []
    for idx in range(len(gt_c2w) - 1):
        gt_rel = np.linalg.inv(gt_c2w[idx]) @ gt_c2w[idx + 1]
        pred_rel = np.linalg.inv(pred_c2w[idx]) @ pred_c2w[idx + 1]
        rel_error = np.linalg.inv(gt_rel) @ pred_rel
        trans_errors.append(translation_error(rel_error))
        rot_errors.append(rotation_error_rad(rel_error))

    return float(np.mean(trans_errors)), float(np.mean(rot_errors))


def compute_ate(gt_c2w: np.ndarray, pred_c2w: np.ndarray) -> float:
    """Compute RMSE Absolute Trajectory Error over aligned camera centers."""
    gt_xyz = gt_c2w[:, :3, 3]
    pred_xyz = pred_c2w[:, :3, 3]
    errors = np.linalg.norm(gt_xyz - pred_xyz, axis=1)
    return float(np.sqrt(np.mean(errors**2)))


def align_trajectory(
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    method: str = "sim3",
) -> tuple[np.ndarray, dict[str, float | list[list[float]]]]:
    """Align predicted c2w poses to GT c2w poses.

    Args:
        pred_c2w: Predicted camera-to-world poses, shape [N, 4, 4].
        gt_c2w: Ground-truth camera-to-world poses, shape [N, 4, 4].
        method: "sim3", "se3", or "none".

    The Sim3/SE3 transform is estimated from camera centers with Umeyama
    alignment, then applied to both predicted centers and predicted rotations.
    """
    method = method.lower()
    if method not in {"sim3", "se3", "none"}:
        raise ValueError(f"Unsupported alignment method: {method}")
    if pred_c2w.shape != gt_c2w.shape:
        raise ValueError(f"Pose shapes differ: pred={pred_c2w.shape}, gt={gt_c2w.shape}")

    aligned = pred_c2w.copy()
    if method == "none":
        return aligned, {"method": method, "scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]}

    with_scale = method == "sim3"
    scale, rotation, translation = umeyama_alignment(
        pred_c2w[:, :3, 3],
        gt_c2w[:, :3, 3],
        with_scale=with_scale,
    )

    # This maps a predicted world coordinate x to the GT world coordinate:
    # x_gt = scale * R_align @ x_pred + t_align.
    aligned[:, :3, :3] = rotation @ pred_c2w[:, :3, :3]
    aligned[:, :3, 3] = scale * (rotation @ pred_c2w[:, :3, 3].T).T + translation

    info = {
        "method": method,
        "scale": float(scale),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
    }
    return aligned, info


def umeyama_alignment(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate similarity transform dst ~= scale * R @ src + t.

    This is a compact NumPy implementation of the Umeyama alignment used for
    trajectory evaluation. It avoids depending on InstantSplat/evo internals.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected src/dst shape [N,3], got {src.shape} and {dst.shape}")
    if len(src) < 2:
        raise ValueError("At least two points are required for trajectory alignment.")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    covariance = (dst_centered.T @ src_centered) / len(src)
    u, singular_values, vh = np.linalg.svd(covariance)

    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        sign[-1] = -1.0
    correction = np.diag(sign)

    rotation = u @ correction @ vh
    if with_scale:
        src_variance = np.mean(np.sum(src_centered**2, axis=1))
        if src_variance <= 1e-12:
            raise ValueError("Predicted camera centers are degenerate; cannot estimate scale.")
        scale = float(np.sum(singular_values * sign) / src_variance)
    else:
        scale = 1.0

    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def plot_trajectories(
    gt_c2w: np.ndarray,
    pred_c2w: np.ndarray,
    image_names: list[str],
    output_path: Path,
    title: str = "Camera Trajectory",
    vid: bool = False,
) -> None:
    """Draw a 3D trajectory plot using evo, matching InstantSplat's style.

    ``pred_c2w`` is expected to be the already aligned prediction used for
    metric computation. InstantSplat aligns once again inside ``plot_pose``;
    here we keep plotting faithful to the metrics to avoid two slightly
    different aligned trajectories in the same report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    traj_ref = PosePath3D(poses_se3=[pose for pose in gt_c2w])
    traj_est_aligned = PosePath3D(poses_se3=[pose for pose in pred_c2w])

    if vid:
        video_dir = output_path.parent / "pose_vid"
        video_dir.mkdir(parents=True, exist_ok=True)
        for pose_idx in range(len(gt_c2w)):
            fig = plt.figure()
            current_ref = PosePath3D(poses_se3=traj_ref.poses_se3[: pose_idx + 1])
            current_est = PosePath3D(poses_se3=traj_est_aligned.poses_se3[: pose_idx + 1])
            ax = fig.add_subplot(111, projection="3d")
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.zaxis.set_tick_params(labelleft=False)
            for style, color, label, trajectory in [
                ("-", "r", "Ours (aligned)", current_est),
                ("--", "b", "Ground-truth", current_ref),
            ]:
                plot.traj(ax, plot.PlotMode.xyz, trajectory, style, color, label)
            ax.view_init(elev=10.0, azim=45.0)
            plt.tight_layout()
            fig.savefig(video_dir / f"pose_vis_{pose_idx:03d}.png")
            plt.close(fig)

    fig = plt.figure(figsize=(8, 7))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelleft=True)
    ax.zaxis.set_tick_params(labelleft=True)

    for style, color, label, trajectory in [
        ("s-", "#2c9e38", "Ours (aligned)", traj_est_aligned),
        ("s-.", "#d12920", "COLMAP (GT)", traj_ref),
    ]:
        plot.traj(ax, plot.PlotMode.xyz, trajectory, style, color, label)

    xyz = np.concatenate([gt_c2w[:, :3, 3], pred_c2w[:, :3, 3]], axis=0)
    # ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=1)
    ax.view_init(elev=30.0, azim=45.0)
    set_axes_equal(ax, xyz)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, transparent=False)
    plt.close(fig)

    names_path = output_path.with_suffix(".names.txt")
    names_path.write_text("\n".join(image_names) + "\n", encoding="utf-8")


def set_axes_equal(ax, xyz: np.ndarray) -> None:
    """Make 3D axes use the same scale so the trajectory shape is not distorted."""
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) * 0.5
    radius = max(float(np.max(maxs - mins)) * 0.5, 1e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
