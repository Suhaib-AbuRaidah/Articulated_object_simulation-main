"""Entry point so the package runs as ``python -m articraft_canon``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
