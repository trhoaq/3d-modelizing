# 3D Reconstruction Pipeline

Few-shot 3D reconstruction from 2-5 images using MASt3R + TSDF + Open3D.

## Overview

This pipeline reconstructs 3D meshes from a few input images using state-of-the-art methods:

1. **Step 1: MASt3R** - Dense matching and depth estimation
2. **Step 2: TSDF** - Clean depth maps and point cloud extraction
3. **Step 3: Mesh** - Poisson Surface Reconstruction or Marching Cubes

## Quick Start

### 1. Setup

```powershell
# Clone repository and install dependencies
.\setup.ps1
```

Or manually:

```bash
git clone --recursive https://github.com/naver/mast3r.git repos/mast3r
cd repos/mast3r
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
cd ../..
pip install -r requirements.txt
```

### 2. Add Images

Add 2-5 images of your scene to the `input/` directory:

```bash
input/
├── image1.jpg
├── image2.jpg
├── image3.jpg
└── ...
```

### 3. Run Pipeline

```bash
# Full pipeline
python main.py --input ./input/ --output ./output/

# Only specific steps
python main.py --input ./input/ --output ./output/ --steps 1 2

# With custom options
python main.py --input ./input/ --output ./output/ \
    --mesh-method poisson \
    --poisson-depth 10 \
    --format obj
```

## Pipeline Steps

### Step 1: MASt3R Reconstruction

Uses [naver/mast3r](https://github.com/naver/mast3r) for dense matching and depth estimation.

```bash
python steps/step1_mast3r.py --input ./input/ --output ./output/step1
```

**Output:**
- `scene.npz` - Depth maps, camera poses, point cloud
- `scene.pt` - MASt3R scene object

### Step 2: TSDF Processing

Uses MASt3R's TSDFPostProcess for clean depth maps.

```bash
python steps/step2_tsdf.py --step1-output ./output/step1 --output ./output/step2
```

**Output:**
- `pointcloud.npz` - Clean point cloud with colors
- `pointcloud.ply` - Point cloud for visualization

### Step 3: Mesh Extraction

Uses Open3D for Poisson Surface Reconstruction or Marching Cubes.

```bash
python steps/step3_mesh.py --step2-output ./output/step2 --output ./output/step3 --format obj
```

**Output:**
- `mesh.obj` - 3D mesh (or .ply/.glb)

## Command Line Options

### Main Pipeline

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | required | Input images directory |
| `--output` | required | Output directory |
| `--steps` | 1 2 3 | Steps to run |
| `--image-size` | 512 | Image resolution |
| `--device` | cuda | Device (cuda/cpu) |
| `--tsdf-thresh` | 0.05 | TSDF threshold |
| `--min-conf` | 2.0 | Minimum confidence |
| `--mesh-method` | poisson | Mesh method |
| `--poisson-depth` | 9 | Poisson depth |
| `--format` | obj | Output format |

### Output Formats

- `obj` - Wavefront OBJ (most compatible)
- `ply` - Polygon File Format
- `glb` - Binary glTF (for web viewers)

## Examples

### Full Pipeline

```bash
python main.py --input ./input/ --output ./output/
```

### MASt3R Only

```bash
python steps/step1_mast3r.py --input ./input/ --output ./output/step1
```

### High-Quality Mesh

```bash
python main.py --input ./input/ --output ./output/ \
    --poisson-depth 12 \
    --format obj
```

### Low-Quality (Faster)

```bash
python main.py --input ./input/ --output ./output/ \
    --poisson-depth 7 \
    --image-size 256
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (recommended)
- 8GB+ RAM

## Project Structure

```
3d-Maker/
├── main.py                    # Main pipeline orchestrator
├── steps/
│   ├── step1_mast3r.py        # MASt3R reconstruction
│   ├── step2_tsdf.py          # TSDF processing
│   └── step3_mesh.py          # Mesh extraction
├── repos/
│   └── mast3r/                # MASt3R repository
├── checkpoints/               # Pretrained weights
├── input/                     # Input images
├── output/                    # Reconstructed mesh
└── examples/                  # Example scripts
```

## Credits

- [MASt3R](https://github.com/naver/mast3r) - Dense matching and 3D reconstruction
- [DUSt3R](https://github.com/naver/dust3r) - Base model for MASt3R
- [Open3D](http://www.open3d.org/) - 3D data processing

## License

This project uses MASt3R which is licensed under CC BY-NC-SA 4.0.
