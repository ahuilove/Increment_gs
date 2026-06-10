from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_read_write_model():
    """Load the project COLMAP writer without being shadowed by eval/utils."""
    module_path = REPO_ROOT / "utils" / "read_write_model.py"
    spec = importlib.util.spec_from_file_location("project_read_write_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load COLMAP reader/writer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rw = load_read_write_model()
Camera = _rw.Camera
Image = _rw.Image
Point3D = _rw.Point3D
rotmat2qvec = _rw.rotmat2qvec
write_model = _rw.write_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LongSplat cameras_all_train/test.json to a COLMAP sparse model for eval/traj_eval.py."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to LongSplat cameras_all_train.json or cameras_all_test.json.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory. If it does not end with sparse/0, the model is written to output/sparse/0.",
    )
    parser.add_argument(
        "--ext",
        choices=[".txt", ".bin"],
        default=".txt",
        help="COLMAP output format. Text is easier to inspect; traj_eval.py supports both.",
    )
    parser.add_argument(
        "--shared-camera",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use one shared camera if all intrinsics and image sizes are identical.",
    )
    parser.add_argument(
        "--copy-images-from",
        default=None,
        help="Optional images directory. If set, matched images are copied/symlinked to output/images.",
    )
    parser.add_argument(
        "--image-ext",
        default=None,
        help="Extension to append when LongSplat image_name has no suffix, e.g. .JPG. "
        "If omitted and --copy-images-from is set, the extension is inferred from source images.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="How to populate output/images when --copy-images-from is set.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing COLMAP model files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    sparse_dir = resolve_sparse_output_dir(Path(args.output).expanduser().resolve())
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    existing = [sparse_dir / f"cameras{args.ext}", sparse_dir / f"images{args.ext}", sparse_dir / f"points3D{args.ext}"]
    if any(path.exists() for path in existing) and not args.overwrite:
        raise FileExistsError(f"{sparse_dir} already contains COLMAP {args.ext} files. Use --overwrite.")

    cameras_json = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(cameras_json, list) or not cameras_json:
        raise ValueError(f"{input_path} must contain a non-empty list of LongSplat cameras.")

    images_src = Path(args.copy_images_from).expanduser().resolve() if args.copy_images_from else None
    image_ext_by_stem = build_image_ext_index(images_src) if images_src else {}
    cameras, images = build_colmap_model(
        cameras_json,
        shared_camera=args.shared_camera,
        image_ext=args.image_ext,
        image_ext_by_stem=image_ext_by_stem,
    )
    sparse_dir.mkdir(parents=True, exist_ok=True)
    write_model(cameras, images, {}, str(sparse_dir), ext=args.ext)

    copied_images = 0
    if args.copy_images_from:
        copied_images = copy_images(
            cameras_json,
            images_src,
            sparse_dir.parent / "images",
            args.copy_mode,
            image_ext=args.image_ext,
            image_ext_by_stem=image_ext_by_stem,
        )

    summary = {
        "input": str(input_path),
        "sparse_dir": str(sparse_dir),
        "format": args.ext,
        "num_cameras": len(cameras),
        "num_images": len(images),
        "shared_camera": bool(args.shared_camera),
        "copied_images": copied_images,
        "pose_convention": "LongSplat JSON stores 3DGS R,T; exported COLMAP uses qvec=rotmat2qvec(R.T), tvec=T.",
    }
    (sparse_dir.parent / "longsplat_json2colmap_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Input LongSplat JSON: {input_path}")
    print(f"Wrote COLMAP sparse model: {sparse_dir}")
    print(f"Format: {args.ext}")
    print(f"Cameras: {len(cameras)}")
    print(f"Images: {len(images)}")
    if args.copy_images_from:
        print(f"Images copied/symlinked: {copied_images}")


def resolve_sparse_output_dir(output: Path) -> Path:
    if output.name == "0" and output.parent.name == "sparse":
        return output
    if output.name == "sparse":
        return output / "0"
    return output / "sparse" / "0"


def build_colmap_model(
    cameras_json: list[dict],
    shared_camera: bool,
    image_ext: str | None,
    image_ext_by_stem: dict[str, str],
) -> tuple[dict[int, Camera], dict[int, Image]]:
    cameras: dict[int, Camera] = {}
    images: dict[int, Image] = {}
    shared_camera_id: int | None = None
    shared_signature: tuple[int, int, float, float, float, float] | None = None

    for idx, cam in enumerate(cameras_json, start=1):
        image_name = image_name_with_extension(cam, image_ext=image_ext, image_ext_by_stem=image_ext_by_stem)
        width = int(round(float(cam["width"])))
        height = int(round(float(cam["height"])))
        fx = float(cam["Focalx"])
        fy = float(cam["Focaly"])
        cx = float(cam.get("cx", width * 0.5))
        cy = float(cam.get("cy", height * 0.5))

        signature = (width, height, fx, fy, cx, cy)
        if shared_camera:
            if shared_signature is None:
                shared_signature = signature
                shared_camera_id = 1
                cameras[shared_camera_id] = Camera(
                    id=shared_camera_id,
                    model="PINHOLE",
                    width=width,
                    height=height,
                    params=np.array([fx, fy, cx, cy], dtype=np.float64),
                )
            elif not intrinsics_close(shared_signature, signature):
                raise ValueError(
                    "Cannot use --shared-camera because LongSplat JSON contains different intrinsics/sizes."
                )
            camera_id = shared_camera_id
        else:
            camera_id = idx
            cameras[camera_id] = Camera(
                id=camera_id,
                model="PINHOLE",
                width=width,
                height=height,
                params=np.array([fx, fy, cx, cy], dtype=np.float64),
            )

        # LongSplat/3DGS cameras store R such that world-to-camera rotation is R.T.
        r_3dgs = np.asarray(cam["R"], dtype=np.float64)
        t_w2c = np.asarray(cam["T"], dtype=np.float64)
        if r_3dgs.shape != (3, 3):
            raise ValueError(f"Camera {idx} has invalid R shape: {r_3dgs.shape}")
        if t_w2c.shape != (3,):
            raise ValueError(f"Camera {idx} has invalid T shape: {t_w2c.shape}")

        r_w2c = r_3dgs.T
        qvec = rotmat2qvec(r_w2c)
        images[idx] = Image(
            id=idx,
            qvec=qvec,
            tvec=t_w2c,
            camera_id=int(camera_id),
            name=image_name,
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )

    return cameras, images


def image_name_with_extension(
    cam: dict,
    image_ext: str | None = None,
    image_ext_by_stem: dict[str, str] | None = None,
) -> str:
    name = str(cam["image_name"])
    suffix = Path(name).suffix
    if suffix:
        return os.path.basename(name)
    stem = os.path.basename(name)
    if image_ext:
        return stem + normalize_ext(image_ext)
    if image_ext_by_stem and stem in image_ext_by_stem:
        return stem + image_ext_by_stem[stem]
    # LongSplat normally stores image_name without extension. PNG is the safest
    # fallback for generated chunks, but pass --image-ext or --copy-images-from
    # when the GT COLMAP model uses .JPG/.jpg names.
    return stem + ".png"


def normalize_ext(ext: str) -> str:
    return ext if ext.startswith(".") else f".{ext}"


def build_image_ext_index(images_src: Path | None) -> dict[str, str]:
    if images_src is None:
        return {}
    allowed = {".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG"}
    index: dict[str, str] = {}
    for path in images_src.iterdir():
        if path.is_file() and path.suffix in allowed:
            index.setdefault(path.stem, path.suffix)
    return index


def intrinsics_close(
    lhs: tuple[int, int, float, float, float, float],
    rhs: tuple[int, int, float, float, float, float],
) -> bool:
    return lhs[:2] == rhs[:2] and np.allclose(lhs[2:], rhs[2:], rtol=1e-8, atol=1e-8)


def copy_images(
    cameras_json: list[dict],
    images_src: Path,
    images_dst: Path,
    mode: str,
    image_ext: str | None,
    image_ext_by_stem: dict[str, str],
) -> int:
    if not images_src.exists():
        raise FileNotFoundError(images_src)
    images_dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for cam in cameras_json:
        target_name = image_name_with_extension(cam, image_ext=image_ext, image_ext_by_stem=image_ext_by_stem)
        source = find_source_image(images_src, cam, target_name)
        target = images_dst / target_name
        if target.exists() or target.is_symlink():
            target.unlink()
        if mode == "copy":
            shutil.copy2(source, target)
        else:
            target.symlink_to(source)
        copied += 1
    return copied


def find_source_image(images_src: Path, cam: dict, target_name: str) -> Path:
    candidates = []
    image_path = cam.get("image_path")
    if image_path:
        candidates.append(Path(str(image_path)))
    candidates.append(images_src / target_name)
    stem = Path(target_name).stem
    for ext in [".png", ".jpg", ".jpeg", ".JPG", ".PNG"]:
        candidates.append(images_src / f"{stem}{ext}")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find source image for {target_name} under {images_src}")


if __name__ == "__main__":
    main()
