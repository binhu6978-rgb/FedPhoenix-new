from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy
import scipy
import torch
import torchvision


def _module_record(module: object) -> dict[str, str]:
    return {
        "version": str(getattr(module, "__version__")),
        "file": str(Path(getattr(module, "__file__")).resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
        "torch": _module_record(torch),
        "torchvision": _module_record(torchvision),
        "numpy": _module_record(numpy),
        "scipy": _module_record(scipy),
        "cuda": {
            "available": torch.cuda.is_available(),
            "torch_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "gpu_capability": (
                list(torch.cuda.get_device_capability(0))
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    (output_dir / "runtime_environment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    (output_dir / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

