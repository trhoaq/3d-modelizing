#!/usr/bin/env python3
"""
3D Reconstruction Pipeline
- Orchestrates Step 1, 2, 3
- Can run individual steps or full pipeline

Usage:
    # Full pipeline
    python main.py --input ./input/ --output ./output/

    # Only specific steps
    python main.py --input ./input/ --output ./output/ --steps 1 2

    # With custom options
    python main.py --input ./input/ --output ./output/ --format obj --mesh-method poisson
"""

import argparse
import sys
import os
import time
from pathlib import Path


def run_pipeline(input_dir: str, output_dir: str,
                 steps: list = [1, 2, 3],
                 image_size: int = 512,
                 device: str = "cuda",
                 tsdf_thresh: float = 0.05,
                 min_conf: float = 2.0,
                 mesh_method: str = "poisson",
                 poisson_depth: int = 9,
                 voxel_size: float = 0.005,
                 output_format: str = "obj",
                 **kwargs):
    """
    Run the 3D reconstruction pipeline

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output
        steps: List of steps to run (1, 2, 3)
        image_size: Image resolution for MASt3R
        device: Device to use (cuda/cpu)
        tsdf_thresh: TSDF threshold
        min_conf: Minimum confidence threshold
        mesh_method: Mesh extraction method (poisson/marching_cubes)
        poisson_depth: Poisson reconstruction depth
        voxel_size: Voxel size for marching cubes
        output_format: Output format (obj/ply/glb)
    """
    print("\n" + "="*60)
    print("  3D Reconstruction Pipeline")
    print("="*60)
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Steps: {steps}")
    print(f"  Device: {device}")
    print("="*60 + "\n")

    # Validate input
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Count images
    image_files = list(input_path.glob("*.jpg")) + \
                  list(input_path.glob("*.jpeg")) + \
                  list(input_path.glob("*.png")) + \
                  list(input_path.glob("*.bmp"))

    if len(image_files) < 2:
        print(f"Error: Need at least 2 images, found {len(image_files)}")
        print(f"Please add images to {input_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images in {input_dir}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Track timing
    start_time = time.time()

    # Step 1: MASt3R Reconstruction
    if 1 in steps:
        from steps.step1_mast3r import run_step1

        step1_output = str(output_path / "step1")
        run_step1(
            input_dir, step1_output,
            image_size=image_size,
            device=device,
            **kwargs
        )

    # Step 2: TSDF Processing
    if 2 in steps:
        from steps.step2_tsdf import run_step2

        step1_output = str(output_path / "step1")
        step2_output = str(output_path / "step2")

        # Check if step 1 output exists
        if not Path(step1_output).exists():
            print(f"Error: Step 1 output not found at {step1_output}")
            print(f"Please run step 1 first")
            sys.exit(1)

        run_step2(
            step1_output, step2_output,
            tsdf_thresh=tsdf_thresh,
            min_conf=min_conf,
            device=device,
            **kwargs
        )

    # Step 3: Mesh Extraction
    if 3 in steps:
        from steps.step3_mesh import run_step3

        step2_output = str(output_path / "step2")
        step3_output = str(output_path / "step3")

        # Check if step 2 output exists
        if not Path(step2_output).exists():
            print(f"Error: Step 2 output not found at {step2_output}")
            print(f"Please run step 2 first")
            sys.exit(1)

        run_step3(
            step2_output, step3_output,
            method=mesh_method,
            poisson_depth=poisson_depth,
            voxel_size=voxel_size,
            output_format=output_format,
            **kwargs
        )

    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("  Pipeline Complete!")
    print("="*60)
    print(f"  Total time: {elapsed_time:.1f} seconds")
    print(f"  Output directory: {output_dir}")
    print(f"  Steps completed: {steps}")

    # List output files
    print(f"\n  Output files:")
    for step in steps:
        step_dir = output_path / f"step{step}"
        if step_dir.exists():
            for f in step_dir.glob("*"):
                if f.is_file():
                    size_kb = f.stat().st_size / 1024
                    print(f"    {f.name} ({size_kb:.1f} KB)")

    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline
    python main.py --input ./input/ --output ./output/

    # Only MASt3R + TSDF (no mesh)
    python main.py --input ./input/ --output ./output/ --steps 1 2

    # Custom mesh options
    python main.py --input ./input/ --output ./output/ \\
        --mesh-method poisson \\
        --poisson-depth 10 \\
        --format obj

    # Run on CPU
    python main.py --input ./input/ --output ./output/ --device cpu
        """
    )

    # Required arguments
    parser.add_argument("--input", required=True,
                       help="Input images directory")
    parser.add_argument("--output", required=True,
                       help="Output directory")

    # Pipeline options
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3],
                       choices=[1, 2, 3],
                       help="Steps to run (default: 1 2 3)")

    # MASt3R options
    parser.add_argument("--image-size", type=int, default=512,
                       help="Image resolution for MASt3R (default: 512)")
    parser.add_argument("--device", default="cuda",
                       help="Device to use (default: cuda)")

    # TSDF options
    parser.add_argument("--tsdf-thresh", type=float, default=0.05,
                       help="TSDF threshold (default: 0.05)")
    parser.add_argument("--min-conf", type=float, default=2.0,
                       help="Minimum confidence threshold (default: 2.0)")

    # Mesh options
    parser.add_argument("--mesh-method", default="poisson",
                       choices=["poisson", "marching_cubes"],
                       help="Mesh extraction method (default: poisson)")
    parser.add_argument("--poisson-depth", type=int, default=9,
                       help="Poisson reconstruction depth (default: 9)")
    parser.add_argument("--voxel-size", type=float, default=0.005,
                       help="Voxel size for marching cubes (default: 0.005)")
    parser.add_argument("--format", default="obj",
                       choices=["obj", "ply", "glb"],
                       help="Output format (default: obj)")

    args = parser.parse_args()

    run_pipeline(
        args.input, args.output,
        args.steps,
        args.image_size,
        args.device,
        args.tsdf_thresh,
        args.min_conf,
        args.mesh_method,
        args.poisson_depth,
        args.voxel_size,
        args.format
    )
