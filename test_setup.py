"""
Quick test script to verify all files and dependencies are in place
"""
import os
import sys

print("="*60)
print("TESTING PROJECT SETUP")
print("="*60)

# Check Python version
print(f"\nPython version: {sys.version}")

# Check required packages
print("\nChecking required packages...")
packages = {
    'numpy': 'np',
    'pandas': 'pd',
    'sklearn': 'sklearn',
    'matplotlib': 'plt'
}

missing = []
for package, alias in packages.items():
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - MISSING!")
        missing.append(package)

if missing:
    print(f"\n  Missing packages: {', '.join(missing)}")
    print("Install with: pip install " + " ".join(missing))
else:
    print("\n✓ All required packages installed!")

# Check files
print("\nChecking project files...")
required_files = {
    'data/credit_card_default.csv': 'Dataset (CSV)',
    'credit_card_default.xls': 'Dataset (Excel)',
    'logistic_regression.py': 'Logistic Regression implementation',
    'Mixed_NB.py': 'Mixed Naive Bayes implementation',
    'Discretized_NB.py': 'Discretized Naive Bayes implementation',
    'main.py': 'Main analysis script'
}

missing_files = []
for file, description in required_files.items():
    if os.path.exists(file):
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} - MISSING!")
        missing_files.append(file)

if missing_files:
    print(f"\n  Missing files: {len(missing_files)}")
else:
    print("\n✓ All required files present!")

# Check figures directory
print("\nChecking figures directory...")
if os.path.exists('figures'):
    print("   figures/ directory exists")
else:
    print("  figures/ directory missing - creating it...")
    os.makedirs('figures', exist_ok=True)
    print("   Created figures/ directory")

print("\n" + "="*60)
if not missing and not missing_files:
    print("✓ SETUP COMPLETE - Ready to run main.py!")
else:
    print(" Please fix the issues above before running main.py")
print("="*60)