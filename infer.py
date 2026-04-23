"""Simple inference script: video/images -> point cloud PLY.

Usage:
    python infer.py --model_path /path/to/ckpt.pt --video_path video.mp4
    python infer.py --model_path /path/to/ckpt.pt --image_folder /path/to/frames/
    python infer.py --model_path /path/to/ckpt.pt --video_path video.mp4 --output_dir /custom/out
"""

import argparse
import os
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from demo import load_images, load_model, postprocess, prepare_for_visualization


def write_ply(path, points, colors):
    """Write colored point cloud to binary PLY."""
    n = len(points)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode()
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    v = np.empty(n, dtype=dtype)
    v["x"], v["y"], v["z"] = points[:, 0], points[:, 1], points[:, 2]
    v["red"], v["green"], v["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(v.tobytes())


def extract_point_cloud(preds, conf_threshold):
    """Flatten world_points filtered by confidence."""
    pts   = preds["world_points"]      # [S, H, W, 3]
    conf  = preds["world_points_conf"] # [S, H, W]
    imgs  = preds["images"]            # [S, 3, H, W] or [S, H, W, 3]

    if imgs.ndim == 4 and imgs.shape[1] == 3:
        imgs = imgs.transpose(0, 2, 3, 1)  # -> [S, H, W, 3]

    mask = conf > conf_threshold
    pts_flat  = pts[mask]
    rgb_float = imgs[mask]
    colors = (rgb_float * 255).clip(0, 255).astype(np.uint8)
    return pts_flat, colors


def main():
    parser = argparse.ArgumentParser(description="LingBot-Map simple inference -> PLY")

    parser.add_argument("--video_path",   type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--fps",          type=int, default=10)
    parser.add_argument("--first_k",      type=int, default=None)
    parser.add_argument("--stride",       type=int, default=1)

    parser.add_argument("--model_path",       type=str, required=True)
    parser.add_argument("--image_size",       type=int, default=518)
    parser.add_argument("--patch_size",       type=int, default=14)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--conf_threshold",   type=float, default=1.5)
    parser.add_argument("--use_sdpa",         action="store_true", default=False)
    parser.add_argument("--output_dir",       type=str, default="results")

    # pass-through args that load_model reads via argparse namespace
    parser.add_argument("--mode",                  type=str,   default="streaming")
    parser.add_argument("--enable_3d_rope",        action="store_true", default=True)
    parser.add_argument("--max_frame_num",         type=int,   default=1024)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--camera_num_iterations", type=int,   default=4)
    parser.add_argument("--offload_to_cpu",        action="store_true", default=False)

    args = parser.parse_args()
    assert args.video_path or args.image_folder, "Provide --video_path or --image_folder"

    # Derive video_id
    if args.video_path:
        video_id = os.path.splitext(os.path.basename(args.video_path))[0]
    else:
        video_id = os.path.basename(args.image_folder.rstrip("/\\"))

    out_dir = os.path.join(args.output_dir, video_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    # Load
    t0 = time.time()
    images, _, _ = load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, first_k=args.first_k, stride=args.stride,
        image_size=args.image_size, patch_size=args.patch_size,
    )
    model = load_model(args, device)
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)
    print(f"Load time: {time.time() - t0:.1f}s")

    images = images.to(device)
    num_frames = images.shape[0]

    keyframe_interval = 1 if num_frames <= 320 else (num_frames + 319) // 320

    # Inference
    print(f"Running inference on {num_frames} frames (dtype={dtype})...")
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model.inference_streaming(
            images,
            num_scale_frames=args.num_scale_frames,
            keyframe_interval=keyframe_interval,
            output_device=torch.device("cpu") if args.offload_to_cpu else None,
        )
    print(f"Inference: {time.time() - t0:.1f}s")

    # Post-process
    predictions, images_cpu = postprocess(predictions, images if not args.offload_to_cpu else predictions["images"])
    vis = prepare_for_visualization(predictions, images_cpu)

    # Save point cloud
    pts, colors = extract_point_cloud(vis, args.conf_threshold)
    print(f"{len(pts)} points (conf > {args.conf_threshold})")

    ply_path = os.path.join(out_dir, "point_cloud.ply")
    write_ply(ply_path, pts, colors)
    print(f"Saved: {ply_path}")

    # Save camera poses
    np.savez(
        os.path.join(out_dir, "camera_poses.npz"),
        extrinsics=vis["extrinsic"],
        intrinsics=vis["intrinsic"],
    )
    print(f"Saved: {os.path.join(out_dir, 'camera_poses.npz')}")


if __name__ == "__main__":
    main()
