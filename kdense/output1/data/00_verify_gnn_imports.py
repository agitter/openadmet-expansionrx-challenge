#!/usr/bin/env python3
"""
Verify GNN package imports for K-Dense ADMET prediction project.
Tests that torch, chemprop, and related dependencies are correctly installed.
"""

import sys
from pathlib import Path

print("=" * 80)
print("GNN Package Import Verification")
print("=" * 80)

# Test imports
packages_to_test = [
    ("torch", "PyTorch"),
    ("chemprop", "Chemprop"),
    ("lightning", "PyTorch Lightning"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("rdkit", "RDKit"),
    ("sklearn", "Scikit-learn"),
]

all_passed = True
versions = {}

for module_name, display_name in packages_to_test:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        versions[display_name] = version
        print(f"✓ {display_name:20s} imported successfully (v{version})")
    except ImportError as e:
        print(f"✗ {display_name:20s} FAILED to import: {e}")
        all_passed = False

print("\n" + "=" * 80)

# Test PyTorch GPU availability
try:
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_devices = torch.cuda.device_count() if cuda_available else 0
    print(f"\nPyTorch CUDA Status:")
    print(f"  - CUDA Available: {cuda_available}")
    print(f"  - CUDA Devices: {cuda_devices}")
    if cuda_available:
        print(f"  - Current Device: {torch.cuda.current_device()}")
        print(f"  - Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("  - Note: Training will use CPU (slower but functional)")
except Exception as e:
    print(f"  - Error checking CUDA: {e}")

print("\n" + "=" * 80)

# Test chemprop CLI availability
try:
    from chemprop import __version__ as chemprop_version
    print(f"\nChemprop Configuration:")
    print(f"  - Version: {chemprop_version}")
    print(f"  - Python API: Available ✓")

    # Check if chemprop CLI commands exist
    try:
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "-m", "chemprop", "--help"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"  - CLI Interface: Available ✓")
        else:
            print(f"  - CLI Interface: Check needed")
    except Exception:
        print(f"  - CLI Interface: Unable to verify")

except Exception as e:
    print(f"\nChemprop check failed: {e}")
    all_passed = False

print("\n" + "=" * 80)

if all_passed:
    print("\n✓ All package imports successful!")
    print("✓ Environment ready for GNN implementation")
    sys.exit(0)
else:
    print("\n✗ Some package imports failed")
    print("✗ Please check installation")
    sys.exit(1)
