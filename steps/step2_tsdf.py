#!/usr/bin/env python3
"""
Step 2: TSDF Processing
- Load scene from Step 1
- Apply TSDF post-processing
- Extract clean point cloud

Uses naver/mast3r repository's TSDFPostProcess directly.
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch


def run_step2(step1_output: str, output_dir: str,
              tsdf_thresh: float = 0.05, clean_depth: bool = True,
              min_conf: float = 2.0, device: str = "cuda"):
    """
    Step 2: TSDF Processing

    Args:
        step1_output: Directory containing Step 1 output
        output_dir: Directory to save output
        tsdf_thresh: TSDF threshold for post-processing
        clean_depth: Whether to clean depth maps
        min_conf: Minimum confidence threshold
        device: Device to use

    Returns:
        points: Point cloud (N, 3)
        colors: Point colors (N, 3)
    """
    print(f"\n{'='*60}")
    print(f"[Step 2] TSDF Processing")
    print(f"{'='*60}")
    print(f"  Input: {step1_output}")
    print(f"  Output: {output_dir}")
    print(f"  TSDF threshold: {tsdf_thresh}")
    print(f"  Min confidence: {min_conf}")

    # Add MASt3R to path
    mast3r_path = os.path.join(os.path.dirname(__file__), '..', 'repos', 'mast3r')
    if mast3r_path not in sys.path:
        sys.path.insert(0, mast3r_path)

    # Import MASt3R modules
    try:
        import mast3r.utils.path_to_dust3r  # noqa: F401
        from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
        from dust3r.utils.device import to_numpy
        has_tsdf = True
    except ImportError as e:
        print(f"  Warning: Could not import MASt3R TSDF: {e}")
        print(f"  Using fallback point cloud extraction")
        has_tsdf = False

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Free GPU memory before starting
    torch.cuda.empty_cache()

    # Try to load scene object for TSDF
    scene_path = Path(step1_output) / "scene.pt"
    scene = None

    if scene_path.exists() and has_tsdf:
        try:
            scene = torch.load(str(scene_path), map_location=device, weights_only=False)
            print(f"  Loaded scene object")

            # Try TSDF post-processing
            print(f"\n  Applying TSDF post-processing...")
            tsdf = TSDFPostProcess(scene, TSDF_thresh=tsdf_thresh)
            pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth, subsample=12))
            colors = to_numpy(scene.imgs)
            print(f"  TSDF processing complete")

        except Exception as e:
            print(f"  Warning: TSDF processing failed: {e}")
            print(f"  Extracting sparse points directly from scene object...")
            torch.cuda.empty_cache()
            scene = None

    # Fallback: extract sparse points from scene object on CPU
    if scene is None and scene_path.exists():
        try:
            scene = torch.load(str(scene_path), map_location='cpu', weights_only=False)
            print(f"  Loaded scene object on CPU")
            from dust3r.utils.device import to_numpy
            pts3d = to_numpy(scene.get_sparse_pts3d())
            colors = [np.array(c) for c in scene.get_pts3d_colors()]
            confs = [np.full(p.shape[0], fill_value=5.0, dtype=float) for p in pts3d]
            print(f"  Extracted sparse points from scene object")

        except Exception as e:
            print(f"  Warning: Scene fallback also failed: {e}")
            raise

    # Ensure pts3d is a list for per-view iteration
    if not isinstance(pts3d, (list, tuple)):
        pts3d = [pts3d[i] for i in range(len(pts3d))]
    if not isinstance(confs, (list, tuple)):
        confs = [confs[i] for i in range(len(confs))]
    if not isinstance(colors, (list, tuple)):
        colors = [colors[i] for i in range(len(colors))]

    # Filter by confidence and extract points/colors
    print(f"\n  Extracting points with confidence > {min_conf}...")
    all_points = []
    all_colors = []

    for i in range(len(pts3d)):
        mask = confs[i] > min_conf
        if mask.any():
            all_points.append(pts3d[i][mask].reshape(-1, 3))
            all_colors.append(colors[i][mask].reshape(-1, 3))

    if all_points:
        points = np.concatenate(all_points)
        colors = np.concatenate(all_colors)
    else:
        print(f"  Warning: No points with confidence > {min_conf}, using all points")
        points = np.concatenate(pts3d).reshape(-1, 3)
        colors = np.concatenate(colors).reshape(-1, 3)

    # Remove invalid points
    print(f"\n  Cleaning point cloud...")
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    colors = colors[valid]

    # Remove outliers
    if len(points) > 1000:
        from scipy.spatial import KDTree
        tree = KDTree(points)
        distances, _ = tree.query(points, k=min(10, len(points) - 1))
        mean_distances = distances.mean(axis=1)
        threshold = np.percentile(mean_distances, 99)
        inlier_mask = mean_distances < threshold
        points = points[inlier_mask]
        colors = colors[inlier_mask]
        print(f"  Removed outliers")

    # Normalize colors to [0, 1]
    if colors.max() > 1.0:
        colors = colors / 255.0

    print(f"  Final point cloud: {len(points)} points")

    # Save point cloud
    output = {
        'points': points,
        'colors': colors,
    }
    np.savez(str(output_path / "pointcloud.npz"), **output)
    print(f"  Saved point cloud to {output_path / 'pointcloud.npz'}")

    # Also save as PLY using trimesh
    try:
        import trimesh
        pcd = trimesh.PointCloud(vertices=points, colors=(colors * 255).astype(np.uint8))
        pcd.export(str(output_path / "pointcloud.ply"))
        print(f"  Saved PLY to {output_path / 'pointcloud.ply'}")
    except Exception as e:
        print(f"  Warning: Could not export PLY: {e}")

    print(f"\n{'='*60}")
    print(f"[Step 2] Complete!")
    print(f"{'='*60}")

    return points, colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2: TSDF Processing"
    )
    parser.add_argument("--step1-output", required=True,
                       help="Directory containing Step 1 output")
    parser.add_argument("--output", required=True,
                       help="Output directory")
    parser.add_argument("--tsdf-thresh", type=float, default=0.05,
                       help="TSDF threshold (default: 0.05)")
    parser.add_argument("--clean-depth", action="store_true", default=True,
                       help="Clean depth maps")
    parser.add_argument("--min-conf", type=float, default=2.0,
                       help="Minimum confidence threshold")
    parser.add_argument("--device", default="cuda",
                       help="Device to use")
    args = parser.parse_args()

    run_step2(
        args.step1_output, args.output,
        args.tsdf_thresh, args.clean_depth,
        args.min_conf, args.device
    )
