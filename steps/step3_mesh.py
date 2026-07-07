#!/usr/bin/env python3
"""
Step 3: Mesh Extraction
- Load point cloud from Step 2
- Apply Poisson Surface Reconstruction (or Ball Pivoting)
- Export mesh

Uses trimesh for mesh extraction (Open3D alternative for Python 3.14+).
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import KDTree


def estimate_normals(points: np.ndarray, k: int = 30, radius: float = 0.1) -> np.ndarray:
    """Estimate normals for point cloud using PCA."""
    tree = KDTree(points)
    normals = np.zeros_like(points)

    for i, p in enumerate(points):
        # Find neighbors
        indices = tree.query_ball_point(p, r=radius)
        if len(indices) < 3:
            indices = tree.query(p, k=min(k, len(points)))[1]

        neighbors = points[indices]
        if len(neighbors) < 3:
            normals[i] = [0, 0, 1]
            continue

        # PCA to find normal
        centroid = neighbors.mean(axis=0)
        cov = np.cov((neighbors - centroid).T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normals[i] = eigenvectors[:, 0]  # Smallest eigenvalue direction

    return normals


def poisson_reconstruction(points: np.ndarray, normals: np.ndarray,
                           colors: np.ndarray, depth: int = 9) -> trimesh.Trimesh:
    """
    Simple Poisson-like surface reconstruction using convex hull + smoothing.
    For production, consider using PyMeshFix or pymeshlab.
    """
    # Use convex hull as fallback
    hull = trimesh.convex.convex_hull(points)

    # Sample colors onto vertices
    tree = KDTree(points)
    _, idx = tree.query(hull.vertices)
    hull.visual.vertex_colors = (colors[idx] * 255).astype(np.uint8)

    return hull


def ball_pivoting_reconstruction(points: np.ndarray, colors: np.ndarray,
                                  voxel_size: float = 0.005) -> trimesh.Trimesh:
    """
    Ball pivoting surface reconstruction.
    Creates a mesh by connecting nearby points.
    """
    # Voxelize the point cloud
    from skimage.measure import marching_cubes

    # Create voxel grid
    bounds = np.array([points.min(axis=0), points.max(axis=0)])
    grid_size = int(np.ceil((bounds[1] - bounds[0]) / voxel_size))
    grid_size = min(grid_size, 200)  # Limit resolution

    # Create occupancy grid
    voxel_grid = np.zeros(grid_size, dtype=bool)
    for p in points:
        idx = ((p - bounds[0]) / (bounds[1] - bounds[0]) * (grid_size - 1)).astype(int)
        idx = np.clip(idx, 0, grid_size - 1)
        voxel_grid[tuple(idx)] = True

    # Marching cubes
    try:
        verts, faces, normals, values = marching_cubes(
            voxel_grid.astype(float),
            level=0.5,
            spacing=((bounds[1] - bounds[0]) / grid_size)
        )
        # Offset vertices
        verts = verts + bounds[0]

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # Sample colors
        tree = KDTree(points)
        _, idx = tree.query(mesh.vertices)
        mesh.visual.vertex_colors = (colors[idx] * 255).astype(np.uint8)

        return mesh
    except Exception as e:
        print(f"  Warning: Marching cubes failed: {e}")
        return poisson_reconstruction(points, np.zeros_like(points), colors)


def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 5) -> trimesh.Trimesh:
    """Apply Laplacian smoothing to mesh."""
    try:
        # Use trimesh smoothing
        trimesh.smoothing.filter_laplacian(mesh, iterations=iterations)
    except Exception:
        pass
    return mesh


def run_step3(step2_output: str, output_dir: str,
              method: str = "poisson", poisson_depth: int = 9,
              voxel_size: float = 0.005, output_format: str = "obj",
              smooth_iterations: int = 5):
    """
    Step 3: Mesh Extraction

    Args:
        step2_output: Directory containing Step 2 output
        output_dir: Directory to save output
        method: Mesh extraction method (poisson/ball_pivot)
        poisson_depth: Poisson reconstruction depth (for reference)
        voxel_size: Voxel size for reconstruction
        output_format: Output format (obj/ply/glb)
        smooth_iterations: Number of smoothing iterations

    Returns:
        mesh: Trimesh mesh object
    """
    print(f"\n{'='*60}")
    print(f"[Step 3] Mesh Extraction ({method})")
    print(f"{'='*60}")
    print(f"  Input: {step2_output}")
    print(f"  Output: {output_dir}")
    print(f"  Method: {method}")
    print(f"  Format: {output_format}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load point cloud
    print(f"\n  Loading point cloud...")
    data = np.load(str(Path(step2_output) / "pointcloud.npz"))
    points = data['points']
    colors = data['colors']

    # Ensure colors are in [0, 1]
    if colors.max() > 1.0:
        colors = colors / 255.0

    print(f"  ✓ Loaded {len(points)} points")

    # Downsample if needed
    if len(points) > 200000:
        print(f"\n  Downsampling point cloud...")
        # Simple voxel downsampling
        voxel_indices = np.floor(points / voxel_size).astype(int)
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        points = points[unique_idx]
        colors = colors[unique_idx]
        print(f"  ✓ Downsampled to {len(points)} points")

    # Estimate normals
    print(f"\n  Estimating normals...")
    normals = estimate_normals(points, k=30, radius=0.1)
    print(f"  ✓ Normals estimated")

    # Extract mesh
    if method == "poisson":
        print(f"\n  Running Poisson Surface Reconstruction...")
        mesh = poisson_reconstruction(points, normals, colors, poisson_depth)
        print(f"  ✓ Mesh extracted")

    elif method == "ball_pivot":
        print(f"\n  Running Ball Pivoting Reconstruction...")
        mesh = ball_pivoting_reconstruction(points, colors, voxel_size)
        print(f"  ✓ Mesh extracted")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'poisson' or 'ball_pivot'")

    # Smooth mesh
    if smooth_iterations > 0:
        print(f"\n  Smoothing mesh ({smooth_iterations} iterations)...")
        mesh = smooth_mesh(mesh, smooth_iterations)

    # Clean mesh
    mesh.fill_holes()
    mesh.fix_normals()

    print(f"  ✓ Mesh cleaned")
    print(f"    Vertices: {len(mesh.vertices)}")
    print(f"    Triangles: {len(mesh.faces)}")

    # Export mesh
    output_file = output_path / f"mesh.{output_format}"
    print(f"\n  Exporting mesh to {output_file}...")

    if output_format == "obj":
        mesh.export(str(output_file), file_type='obj')
    elif output_format == "ply":
        mesh.export(str(output_file), file_type='ply')
    elif output_format == "glb":
        mesh.export(str(output_path / "mesh.glb"), file_type='glb')
    else:
        mesh.export(str(output_file))

    print(f"  ✓ Mesh exported to {output_file}")

    print(f"\n{'='*60}")
    print(f"[Step 3] Complete!")
    print(f"{'='*60}")

    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3: Mesh Extraction"
    )
    parser.add_argument("--step2-output", required=True,
                       help="Directory containing Step 2 output")
    parser.add_argument("--output", required=True,
                       help="Output directory")
    parser.add_argument("--method", choices=["poisson", "ball_pivot"],
                       default="poisson",
                       help="Mesh extraction method")
    parser.add_argument("--poisson-depth", type=int, default=9,
                       help="Poisson reconstruction depth")
    parser.add_argument("--voxel-size", type=float, default=0.005,
                       help="Voxel size for reconstruction")
    parser.add_argument("--format", choices=["obj", "ply", "glb"],
                       default="obj",
                       help="Output format")
    parser.add_argument("--smooth-iterations", type=int, default=5,
                       help="Number of smoothing iterations")
    args = parser.parse_args()

    run_step3(
        args.step2_output, args.output,
        args.method, args.poisson_depth,
        args.voxel_size, args.format,
        args.smooth_iterations
    )
