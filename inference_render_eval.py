"""LingBot-Map inference + render pipeline for a single video directory.

Processes each clip_* subdirectory using LingBot-Map streaming inference,
renders three viewpoints, and saves a .ply point cloud for evaluation.

Usage:
    python inference_render_eval.py <input_video_dir> <output_dir> \
        --model_path /path/to/lingbot-map-long.pt [--force]

    input_video_dir : directory containing clip_* subdirs with frame images
    output_dir      : base output dir; per-clip results go in
                      <output_dir>/<video_stem>/clip_*/
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def render_from_camera(points, colors, extrinsics, intrinsics,
                       width=518, height=518, point_size=2):
    """Project point cloud onto an image plane using a z-buffer rasterizer."""
    world_to_cam = np.linalg.inv(extrinsics)
    points_h = np.hstack([points, np.ones((len(points), 1))])
    points_cam = (world_to_cam @ points_h.T).T[:, :3]

    valid = points_cam[:, 2] > 0.01
    points_cam = points_cam[valid]
    colors_valid = colors[valid]

    if len(points_cam) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_proj = (points_cam[:, 0] * fx / points_cam[:, 2]) + cx
    y_proj = (points_cam[:, 1] * fy / points_cam[:, 2]) + cy

    in_bounds = (x_proj >= 0) & (x_proj < width) & (y_proj >= 0) & (y_proj < height)
    x_coords = x_proj[in_bounds].astype(int)
    y_coords = y_proj[in_bounds].astype(int)
    colors_vis = colors_valid[in_bounds]
    depths = points_cam[in_bounds, 2]

    if len(x_coords) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    order = np.argsort(-depths)
    x_coords = x_coords[order]
    y_coords = y_coords[order]
    colors_vis = colors_vis[order]

    offsets = np.arange(-point_size, point_size + 1)
    dx, dy = np.meshgrid(offsets, offsets)
    dx, dy = dx.flatten(), dy.flatten()
    k = len(dx)

    px = (x_coords[:, None] + dx[None, :]).flatten()
    py = (y_coords[:, None] + dy[None, :]).flatten()
    c  = np.repeat(colors_vis, k, axis=0)

    valid_splat = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[py[valid_splat], px[valid_splat]] = c[valid_splat]
    return image


def build_45_camera(ext_orig, intr_orig, center):
    """Elevated 45° view camera."""
    pos = ext_orig[:3, 3]
    fwd = ext_orig[:3, 2]
    horiz_dist = np.linalg.norm((center - pos)[:2])
    elev_h = horiz_dist * np.tan(np.radians(45))

    fwd_h = fwd.copy()
    fwd_h[1] = 0
    n = np.linalg.norm(fwd_h)
    if n > 1e-6:
        fwd_h /= n

    ext = ext_orig.copy()
    ext[1, 3] += elev_h
    ext[:3, 3] -= fwd_h * elev_h

    intr = intr_orig.copy()
    intr[0, 0] *= 0.7
    intr[1, 1] *= 0.7
    return ext, intr


def build_110_camera(ext_orig, intr_orig, center):
    """Steep overhead 110° view camera."""
    pos = ext_orig[:3, 3]
    fwd = ext_orig[:3, 2]
    horiz_dist = np.linalg.norm((center - pos)[:2])
    vert_off = horiz_dist * np.tan(np.radians(110))

    fwd_h = fwd.copy()
    fwd_h[1] = 0
    n = np.linalg.norm(fwd_h)
    if n > 1e-6:
        fwd_h /= n

    new_pos = center.copy()
    new_pos[1] += abs(vert_off) * 1.5
    new_pos -= fwd_h * horiz_dist * 2.0

    right = ext_orig[:3, 0].copy()
    look = center - new_pos
    look /= np.linalg.norm(look)
    up = np.cross(look, right)
    n = np.linalg.norm(up)
    if n > 1e-6:
        up /= n
    right = np.cross(up, look)

    ext = ext_orig.copy()
    ext[:3, 0] = right
    ext[:3, 1] = up
    ext[:3, 2] = look
    ext[:3, 3] = new_pos

    intr = intr_orig.copy()
    intr[0, 0] *= 0.08
    intr[1, 1] *= 0.08
    return ext, intr


# ---------------------------------------------------------------------------
# Point cloud helpers
# ---------------------------------------------------------------------------

def filter_point_cloud(world_points, confidences, images_rgb, conf_threshold=0.5):
    """Flatten, filter by confidence, flip Y/Z to match render coord system."""
    pts_flat  = world_points.reshape(-1, 3)
    conf_flat = confidences.reshape(-1)
    rgb_flat  = images_rgb.reshape(-1, 3)

    mask = conf_flat > conf_threshold
    pts = pts_flat[mask].copy()
    pts[:, 1] *= -1
    pts[:, 2] *= -1
    colors = (rgb_flat[mask] * 255).clip(0, 255).astype(np.uint8)
    return pts, colors


def save_ply(path, points, colors):
    """Write colored point cloud to a binary-little-endian PLY file."""
    n = len(points)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    vertex = np.empty(n, dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"]   = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"]  = colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(vertex.tobytes())


def render_views(points, colors, extrinsics_4x4, intrinsics):
    """Render ground, 45°, and 110° views from the first camera pose."""
    cam_poses = extrinsics_4x4.copy()
    cam_poses[:, 1, :] *= -1
    cam_poses[:, 2, :] *= -1

    center = points.mean(axis=0)
    ext0  = cam_poses[0]
    intr0 = intrinsics[0]

    views = {
        "ground_0deg":    (ext0, intr0),
        "elevated_45deg":  build_45_camera(ext0, intr0, center),
        "elevated_110deg": build_110_camera(ext0, intr0, center),
    }

    renders = {}
    for name, (ext, intr) in views.items():
        renders[name] = render_from_camera(points, colors, ext, intr, point_size=2)
    return renders


# ---------------------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------------------

def load_model(model_path, device, num_scale_frames=8,
               kv_cache_sliding_window=64, camera_num_iterations=4,
               use_sdpa=False):
    from lingbot_map.models.gct_stream import GCTStream

    model = GCTStream(
        img_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=kv_cache_sliding_window,
        kv_cache_scale_frames=num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=use_sdpa,
        camera_num_iterations=camera_num_iterations,
    )

    print(f"Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys : {len(missing)}")
    if unexpected:
        print(f"  Unexpected   : {len(unexpected)}")
    return model.to(device).eval()


def run_inference(model, image_paths, device, dtype, num_scale_frames=8):
    """Load frames, run streaming inference, return numpy predictions."""
    images = load_and_preprocess_images(
        [str(p) for p in image_paths],
        mode="crop", image_size=518, patch_size=14,
    ).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        preds = model.inference_streaming(
            images,
            num_scale_frames=num_scale_frames,
            keyframe_interval=1,
            output_device=torch.device("cpu"),
        )

    # pose_enc → w2c 3x4 → c2w 4x4
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        preds["pose_enc"], images.shape[-2:]
    )
    S = extrinsic.shape[0]
    ext_4x4 = torch.zeros(S, 4, 4, device=extrinsic.device, dtype=extrinsic.dtype)
    ext_4x4[:, :3, :4] = extrinsic
    ext_4x4[:, 3, 3] = 1.0
    ext_4x4 = closed_form_inverse_se3_general(ext_4x4)

    return {
        "extrinsic":        ext_4x4.cpu().numpy(),                          # [S,4,4] c2w
        "intrinsic":        intrinsic.cpu().numpy(),                        # [S,3,3]
        "world_points":     preds["world_points"].squeeze(0).cpu().numpy(), # [S,H,W,3]
        "world_points_conf": preds["world_points_conf"].squeeze(0).cpu().numpy(), # [S,H,W]
        "images_rgb":       images.permute(0, 2, 3, 1).cpu().numpy(),      # [S,H,W,3]
    }


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------

RENDER_FILES = ["ground_0deg.png", "elevated_45deg.png", "elevated_110deg.png"]


def is_clip_done(clip_out_dir: Path) -> bool:
    return all((clip_out_dir / f).exists() for f in RENDER_FILES)


def process_clip(model, clip_dir: Path, clip_out_dir: Path,
                 device, dtype, num_scale_frames, conf_threshold, force):
    image_files = sorted(
        list(clip_dir.glob("*.jpg")) + list(clip_dir.glob("*.png"))
    )
    if not image_files:
        print(f"  [WARN] No images in {clip_dir}")
        return False, 0

    if not force and is_clip_done(clip_out_dir):
        print(f"  [SKIP] {clip_dir.name} — renders already exist")
        return True, 0

    print(f"  {len(image_files)} frames — running inference...")
    preds = run_inference(model, image_files, device, dtype, num_scale_frames)

    clip_out_dir.mkdir(parents=True, exist_ok=True)

    # ---- point cloud ----
    pts, colors = filter_point_cloud(
        preds["world_points"], preds["world_points_conf"],
        preds["images_rgb"], conf_threshold,
    )
    print(f"  {len(pts)} points (conf > {conf_threshold})")

    ply_path = clip_out_dir / "point_cloud.ply"
    save_ply(ply_path, pts, colors)
    print(f"  [OK] {ply_path.name}")

    # ---- rendered views ----
    renders = render_views(pts, colors, preds["extrinsic"], preds["intrinsic"])
    for name, img_arr in renders.items():
        out = clip_out_dir / f"{name}.png"
        Image.fromarray(img_arr).save(out)
        print(f"  [OK] {out.name}")

    # ---- camera poses ----
    cam_data = {
        "num_cameras": len(image_files),
        "cameras": [
            {
                "frame_id": i,
                "image_path": str(image_files[i]),
                "extrinsics": preds["extrinsic"][i].tolist(),
                "intrinsics": preds["intrinsic"][i].tolist(),
            }
            for i in range(len(image_files))
        ],
    }
    with open(clip_out_dir / "camera_poses.json", "w") as f:
        json.dump(cam_data, f, indent=2)

    np.savez_compressed(
        clip_out_dir / "camera_poses.npz",
        extrinsics=preds["extrinsic"],
        intrinsics=preds["intrinsic"],
        image_paths=np.array([str(p) for p in image_files]),
    )

    return True, len(pts)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = ["video_id", "clip_id", "status", "num_points", "last_updated"]


def read_csv(csv_path: Path):
    rows = {}
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                rows[(row["video_id"], row["clip_id"])] = row
    return rows


def write_csv(csv_path: Path, rows: dict):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def update_csv(csv_path, video_id, clip_id, status, num_points=0):
    rows = read_csv(csv_path)
    rows[(video_id, clip_id)] = {
        "video_id":    video_id,
        "clip_id":     clip_id,
        "status":      status,
        "num_points":  num_points,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_csv(csv_path, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LingBot-Map inference + render (single video eval)"
    )
    parser.add_argument("input_dir",  type=Path, help="Video dir with clip_* subdirs")
    parser.add_argument("output_dir", type=Path, help="Base output directory")
    parser.add_argument("--model_path",       type=str,   required=True)
    parser.add_argument("--progress_csv",     type=Path,  default=None,
                        help="CSV for per-clip progress (default: <output_dir>/progress.csv)")
    parser.add_argument("--conf_threshold",   type=float, default=0.5)
    parser.add_argument("--num_scale_frames", type=int,   default=8)
    parser.add_argument("--use_sdpa",         action="store_true", default=False)
    parser.add_argument("--force",            action="store_true",
                        help="Reprocess clips that already have outputs")
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"[ERROR] input_dir does not exist: {args.input_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.progress_csv or args.output_dir / "progress.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    print(f"Device : {device}  dtype: {dtype}")
    print(f"Input  : {args.input_dir}")
    print(f"Output : {args.output_dir}")
    print(f"CSV    : {csv_path}")

    print("\nLoading model...")
    model = load_model(
        args.model_path, device,
        num_scale_frames=args.num_scale_frames,
        use_sdpa=args.use_sdpa,
    )
    if dtype != torch.float32 and hasattr(model, "aggregator"):
        model.aggregator = model.aggregator.to(dtype=dtype)
    print("[OK] Model loaded\n")

    video_id  = args.input_dir.name
    clip_dirs = sorted(
        d for d in args.input_dir.iterdir()
        if d.is_dir() and d.name.startswith("clip_")
    )

    if not clip_dirs:
        print(f"[WARN] No clip_* dirs found in {args.input_dir}")
        sys.exit(0)

    print(f"Video : {video_id}  ({len(clip_dirs)} clips)")
    print("=" * 60)

    success = failed = 0
    for clip_dir in clip_dirs:
        clip_id   = clip_dir.name
        clip_out  = args.output_dir / video_id / clip_id
        print(f"\n[{clip_id}]")
        try:
            ok, n_pts = process_clip(
                model, clip_dir, clip_out, device, dtype,
                args.num_scale_frames, args.conf_threshold, args.force,
            )
            status = "complete" if ok else "skipped"
            update_csv(csv_path, video_id, clip_id, status, n_pts)
            success += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [ERROR] {e}")
            update_csv(csv_path, video_id, clip_id, "failed")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Video        : {video_id}")
    print(f"Clips done   : {success}")
    print(f"Clips failed : {failed}")
    print(f"Progress CSV : {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
