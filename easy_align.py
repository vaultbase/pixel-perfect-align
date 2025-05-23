#!/usr/bin/env python3
"""
Easy alignment script - just pass a folder path!
Usage: python easy_align.py /path/to/images
"""

import sys
from pathlib import Path
import subprocess


def main():
    if len(sys.argv) < 2:
        print("Usage: python easy_align.py /path/to/images")
        print("\nThis will automatically:")
        print("  - Find all images in the folder")
        print("  - Create an 'Aligned' subfolder")
        print("  - Export transforms and composite")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        sys.exit(1)
    
    if not input_path.is_dir():
        print(f"Error: Path is not a directory: {input_path}")
        sys.exit(1)
    
    # Run the main alignment script
    cmd = [
        sys.executable,
        "align.py",
        "--input-dir", str(input_path)
    ]
    
    print(f"Starting alignment for: {input_path}")
    print("Output will be saved to: {input_path}/Aligned")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError during alignment: {e}")
        sys.exit(1)
    
    print("\n✅ Alignment completed!")
    print(f"Check the results in: {input_path}/Aligned")


if __name__ == "__main__":
    main()