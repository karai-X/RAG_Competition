from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _to_rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _run(command: list[str], label: str) -> None:
    print("[run]", label)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create virtual environment and install dependencies")
    parser.add_argument("--venv-dir", type=Path, default=Path(".venv"), help="Virtual environment directory")
    parser.add_argument(
        "--recreate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete and recreate venv if it exists",
    )
    args = parser.parse_args()

    manual_root = Path(__file__).resolve().parents[1]
    venv_dir = (manual_root / args.venv_dir).resolve()
    rel_venv = _to_rel(venv_dir, manual_root)

    if args.recreate and venv_dir.exists():
        shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        _run([sys.executable, "-m", "venv", str(venv_dir)], f"python -m venv {rel_venv}")

    py = _venv_python(venv_dir)
    requirements = manual_root / "requirements.txt"
    rel_requirements = _to_rel(requirements, manual_root)

    _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], "pip install --upgrade pip setuptools wheel")
    _run([str(py), "-m", "pip", "install", "-r", str(requirements)], f"pip install -r {rel_requirements}")

    print("Environment setup completed.")
    print(f"Virtual environment: {rel_venv}")
    print("Train command: python scripts/run_train.py")


if __name__ == "__main__":
    main()
