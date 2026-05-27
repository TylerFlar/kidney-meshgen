import blenderproc as bproc  # noqa: F401
# ruff: noqa: I001

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kidney_meshgen.blenderproc_render import main


if __name__ == "__main__":
    main()
