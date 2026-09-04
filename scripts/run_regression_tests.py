from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

# Optional read-only fallback for this machine's currently incomplete user-site
# installation.  Normal runs leave these variables unset and use fd unchanged.
_FALLBACK_SITE = os.environ.get("FEDRAD_FALLBACK_SITE_PACKAGES")
_FALLBACK_DLLS = os.environ.get("FEDRAD_FALLBACK_DLL_DIRS", "")
if _FALLBACK_SITE:
    sys.path.insert(0, _FALLBACK_SITE)
if os.name == "nt":
    for _directory in filter(None, _FALLBACK_DLLS.split(os.pathsep)):
        os.add_dll_directory(_directory)

# The fd environment keeps PyTorch in the Python user site while NumPy is in
# the Conda environment.  Import NumPy before PyTorch's DataLoader bootstrap so
# the canonical Conda module is fully initialized rather than observed through
# a partially initialized user-site import.
import numpy  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(PROJECT_ROOT / "tests"),
        top_level_dir=str(PROJECT_ROOT),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
