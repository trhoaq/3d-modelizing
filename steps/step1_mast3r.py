#!/usr/bin/env python3
"""
Step 1: MASt3R Reconstruction
- Load images
- Run sparse global alignment
- Extract depth maps, camera poses, point cloud

Uses naver/mast3r repository directly.
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch


def run_step1(input_dir: str, output_dir: str,
              image_size: int = 512, device: str = "cuda",
              niter1: int = 500, niter2: int = 200,
              matching_conf_thr: float = 5.0):
    """
    Step 1: MASt3R Reconstruction

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output
        image_size: Image resolution for MASt3R
        device: Device to use (cuda/cpu)
        niter1: Number of iterations for first optimization
        niter2: Number of iterations for second optimization
        matching_conf_thr: Matching confidence threshold

    Returns:
        scene: MASt3R scene object
    """
    print(f"\n{'='*60}")
    print(f"[Step 1] MASt3R Reconstruction")
    print(f"{'='*60}")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Image size: {image_size}")
    print(f"  Device: {device}")

    # Add MASt3R to path
    mast3r_path = os.path.join(os.path.dirname(__file__), '..', 'repos', 'mast3r')
    if mast3r_path not in sys.path:
        sys.path.insert(0, mast3r_path)

    # Import MASt3R modules
    try:
        import mast3r.utils.path_to_dust3r  # noqa: F401
        from mast3r.model import AsymmetricMASt3R
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        from dust3r.utils.image import load_images
        from dust3r.utils.device import to_numpy
        from dust3r.image_pairs import make_pairs
    except ImportError as e:
        print(f"  Error importing MASt3R: {e}")
        print(f"  Please run setup.ps1 first")
        raise

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\n  Loading MASt3R model...")
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    
    # Use float16 to save memory if on CPU or limited RAM
    if device == "cpu":
        model = model.half()
        print(f"  ✓ Model loaded (float16 for memory optimization)")
    else:
        print(f"  ✓ Model loaded")

    # Load images
    print(f"\n  Loading images...")
    input_path = Path(input_dir)
    image_files = sorted(
        [str(x) for x in input_path.iterdir()
         if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    )

    if len(image_files) < 2:
        raise ValueError(f"Need at least 2 images, got {len(image_files)}")

    print(f"  Found {len(image_files)} images")
    images = load_images(image_files, size=image_size)
    print(f"  ✓ Images loaded")

    # Generate pairs
    print(f"\n  Generating image pairs...")
    pairs = make_pairs(images, scene_graph='complete',
                       prefilter=None, symmetrize=True)
    print(f"  ✓ Generated {len(pairs)} pairs")

    # Run sparse global alignment
    print(f"\n  Running sparse global alignment...")
    cache_dir = str(output_path / "cache")
    os.makedirs(cache_dir, exist_ok=True)

    scene = sparse_global_alignment(
        image_files, pairs, cache_dir,
        model,
        lr1=0.07, niter1=niter1,
        lr2=0.014, niter2=niter2,
        device=device,
        opt_depth=True,
        shared_intrinsics=False,
        matching_conf_thr=matching_conf_thr
    )
    print(f"  ✓ Sparse global alignment complete")

    # Extract scene data
    print(f"\n  Extracting scene data...")
    scene_data = {
        'imgs': to_numpy(scene.imgs),
        'pts3d': to_numpy(scene.get_pts3d()),
        'focals': to_numpy(scene.get_focals()),
        'poses': to_numpy(scene.get_im_poses()),
        'confidences': to_numpy(scene.get_masks()),
        'image_files': image_files,
    }
    np.savez(str(output_path / "scene.npz"), **scene_data)
    print(f"  ✓ Saved scene data to {output_path / 'scene.npz'}")

    # Save scene object for later use
    torch.save(scene, str(output_path / "scene.pt"))
    print(f"  ✓ Saved scene object to {output_path / 'scene.pt'}")

    print(f"\n{'='*60}")
    print(f"[Step 1] Complete!")
    print(f"{'='*60}")

    return scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: MASt3R Reconstruction"
    )
    parser.add_argument("--input", required=True,
                       help="Input images directory")
    parser.add_argument("--output", required=True,
                       help="Output directory")
    parser.add_argument("--image-size", type=int, default=512,
                       help="Image resolution (default: 512)")
    parser.add_argument("--device", default="cuda",
                       help="Device to use (default: cuda)")
    parser.add_argument("--niter1", type=int, default=500,
                       help="Iterations for first optimization")
    parser.add_argument("--niter2", type=int, default=200,
                       help="Iterations for second optimization")
    parser.add_argument("--matching-conf-thr", type=float, default=5.0,
                       help="Matching confidence threshold")
    args = parser.parse_args()

    run_step1(
        args.input, args.output,
        args.image_size, args.device,
        args.niter1, args.niter2,
        args.matching_conf_thr
    )
