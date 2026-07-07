#!/usr/bin/env python3
"""
Example: Run only Step 2 (TSDF Processing)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from steps.step2_tsdf import run_step2

if __name__ == "__main__":
    run_step2(
        step1_output="./output/step1",
        output_dir="./output/step2",
        tsdf_thresh=0.05,
        min_conf=2.0,
        device="cuda"
    )
