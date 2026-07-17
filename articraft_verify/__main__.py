"""Entry point so the package runs as ``python -m articraft_verify``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
