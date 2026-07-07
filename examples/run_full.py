#!/usr/bin/env python3
"""
Example: Run full pipeline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import run_pipeline

if __name__ == "__main__":
    run_pipeline(
        input_dir="./input",
        output_dir="./output",
        steps=[1, 2, 3],
        image_size=512,
        device="cuda",
        tsdf_thresh=0.05,
        min_conf=2.0,
        mesh_method="poisson",
        poisson_depth=9,
        output_format="obj"
    )
