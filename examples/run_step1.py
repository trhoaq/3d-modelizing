#!/usr/bin/env python3
"""
Example: Run only Step 1 (MASt3R Reconstruction)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from steps.step1_mast3r import run_step1

if __name__ == "__main__":
    run_step1(
        input_dir="./input",
        output_dir="./output/step1",
        image_size=512,
        device="cuda"
    )
