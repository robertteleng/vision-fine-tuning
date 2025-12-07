#!/usr/bin/env python3
"""
Build distribution package for Windows.

Creates a clean dist/ folder with only the files needed to run the app.

Usage:
    python scripts/build_dist.py
    python scripts/build_dist.py --zip  # Also create ZIP file
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DIST_DIR = PROJECT_ROOT / "dist" / "VR-Pillar-Detector"

# Files to include in distribution
DIST_FILES = [
    # Core app
    ("app.py", "app.py"),
    ("requirements.txt", "requirements.txt"),
    ("config.yaml", "config.yaml"),

    # Windows scripts
    ("templates/install.bat", "install.bat"),
    ("templates/run.bat", "run.bat"),

    # Data config
    ("data/pillar.yaml", "data/pillar.yaml"),
]

# Folders to include
DIST_FOLDERS = [
    ("models", "models", ["*.pt"]),  # Only .pt files (portable)
]


def clean_dist():
    """Remove existing dist folder."""
    if DIST_DIR.exists():
        print(f"Cleaning {DIST_DIR}...")
        shutil.rmtree(DIST_DIR)


def create_dist():
    """Create distribution folder with required files."""
    print(f"Creating distribution in {DIST_DIR}...")
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    # Copy individual files
    for src, dst in DIST_FILES:
        src_path = PROJECT_ROOT / src
        dst_path = DIST_DIR / dst

        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            print(f"  + {dst}")
        else:
            print(f"  ! {src} not found, skipping")

    # Copy folders with patterns
    for src_folder, dst_folder, patterns in DIST_FOLDERS:
        src_path = PROJECT_ROOT / src_folder
        dst_path = DIST_DIR / dst_folder

        if src_path.exists():
            dst_path.mkdir(parents=True, exist_ok=True)
            for pattern in patterns:
                for file in src_path.glob(pattern):
                    shutil.copy2(file, dst_path / file.name)
                    print(f"  + {dst_folder}/{file.name}")
        else:
            print(f"  ! {src_folder}/ not found, skipping")


def create_zip():
    """Create ZIP file from dist folder."""
    zip_path = PROJECT_ROOT / "dist" / "VR-Pillar-Detector"
    print(f"Creating {zip_path}.zip...")
    shutil.make_archive(str(zip_path), "zip", DIST_DIR.parent, "VR-Pillar-Detector")

    # Show size
    zip_file = Path(f"{zip_path}.zip")
    size_mb = zip_file.stat().st_size / (1024 * 1024)
    print(f"  Created: {zip_file.name} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build distribution package")
    parser.add_argument("--zip", action="store_true", help="Also create ZIP file")
    args = parser.parse_args()

    print("=" * 50)
    print("  VR Pillar Detector - Build Distribution")
    print("=" * 50)
    print()

    clean_dist()
    create_dist()

    if args.zip:
        print()
        create_zip()

    print()
    print("=" * 50)
    print("  Done!")
    print("=" * 50)
    print()
    print(f"Distribution folder: {DIST_DIR}")
    if args.zip:
        print(f"ZIP file: {DIST_DIR}.zip")


if __name__ == "__main__":
    main()
