import importlib.metadata
import platform
import shutil
import subprocess
import sys


EXPECTED = {
    "jittor": "1.3.10.0",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "scipy": "1.13.1",
    "scikit-learn": "1.5.2",
    "xgboost": "2.1.4",
}


def main():
    print(f"platform={platform.platform()}")
    print(f"python={platform.python_version()}")
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("Python 3.10 is required by the audit environment")

    mismatches = []
    for package, expected in EXPECTED.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        print(f"{package}={actual} (expected {expected})")
        if actual != expected:
            mismatches.append(f"{package}: expected {expected}, found {actual}")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        print("gpu=" + (result.stdout.strip() or "nvidia-smi query failed"))
    else:
        print("gpu=nvidia-smi not found; XGBoost will fall back to CPU")

    if mismatches:
        raise SystemExit("Dependency check failed:\n- " + "\n- ".join(mismatches))
    print("Environment check passed.")


if __name__ == "__main__":
    main()
