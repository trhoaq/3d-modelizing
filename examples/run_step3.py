#!/usr/bin/env python3
"""
Example: Run only Step 3 (Mesh Extraction)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from steps.step3_mesh import run_step3

if __name__ == "__main__":
    run_step3(
        step2_output="./output/step2",
        output_dir="./output/step3",
        method="poisson",
        poisson_depth=9,
        voxel_size=0.005,
        output_format="obj",
        smooth_iterations=5
    )
