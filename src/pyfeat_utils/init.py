from __future__ import annotations

import sys

from pyfeat_utils.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["init", *sys.argv[1:]]))
