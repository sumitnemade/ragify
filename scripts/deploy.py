#!/usr/bin/env python3
"""
Deployment script for Ragify plugin.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result


def clean_build():
    """Clean previous build artifacts."""
    print("🧹 Cleaning build artifacts...")
    dirs_to_clean = ["build", "dist", "*.egg-info"]
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"   Removed {dir_name}")


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    run_command("python -m pytest tests/ -v --cov=ragify")


def build_package():
    """Build the package."""
    print("📦 Building package...")
    run_command("python -m build")


def check_package():
    """Check the built package."""
    print("🔍 Checking package...")
    run_command("python -m twine check dist/*")


def main():
    """Main deployment process."""
    print("🚀 Starting Ragify deployment process...")
    print("=" * 50)
    
    # Ensure we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("Error: pyproject.toml not found. Please run from the project root.")
        sys.exit(1)
    
    # Clean previous builds
    clean_build()
    
    # Run tests
    run_tests()
    
    # Build package
    build_package()
    
    # Check package
    check_package()
    
    print("✅ Deployment preparation complete!")
    print("\n📋 Next steps:")
    print("1. Review the built package in dist/")
    print("2. Test installation: pip install dist/ragify-*.whl")
    print("3. Upload to PyPI: python -m twine upload dist/*")
    print("4. Create a GitHub release")


if __name__ == "__main__":
    main()
