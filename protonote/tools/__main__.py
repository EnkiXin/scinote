"""Entry-point so `python -m protonote.tools --selftest` works."""
from __future__ import annotations

import argparse

from protonote.tools import _selftest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print("usage: python -m protonote.tools --selftest")


if __name__ == "__main__":
    main()
