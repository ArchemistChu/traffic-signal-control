#!/usr/bin/env python3
"""
Safe cleanup utility for generated files.

Default behavior is DRY-RUN (no deletion). Use --apply to actually delete.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PATTERNS = [
    "__pycache__/",
    "**/__pycache__/",
    "*.pyc",
    "temp_*.py",
    "temp_*.json",
    "output/web_results/**/*.status.json",
    "output/web_results/**/*.tmp",
]


def iter_matches(root: Path) -> list[Path]:
    matches: list[Path] = []
    for pattern in PATTERNS:
        for p in root.glob(pattern):
            matches.append(p)
    # unique + stable order
    return sorted({p.resolve() for p in matches}, key=lambda x: str(x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually delete matched files/folders.")
    parser.add_argument("--root", type=str, default=".", help="Project root path.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    paths = iter_matches(root)

    print(f"Root: {root}")
    print(f"Matched: {len(paths)}")
    for p in paths:
        print(f" - {p}")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to delete.")
        return

    deleted = 0
    for p in paths:
        try:
            if p.is_dir():
                for sub in sorted(p.rglob("*"), reverse=True):
                    if sub.is_file():
                        sub.unlink(missing_ok=True)
                    elif sub.is_dir():
                        sub.rmdir()
                p.rmdir()
            else:
                p.unlink(missing_ok=True)
            deleted += 1
        except Exception as exc:
            print(f" ! Failed to delete {p}: {exc}")

    print(f"\nDeleted: {deleted}/{len(paths)}")


if __name__ == "__main__":
    main()

