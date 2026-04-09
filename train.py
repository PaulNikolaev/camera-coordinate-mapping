"""Thin CLI wrapper for baseline model training."""

from __future__ import annotations

import sys


def main() -> int:
    """Import and run the reusable training CLI."""
    try:
        from solution.train import main as solution_main
    except ImportError:
        print(
            "Training dependencies are missing. Install them with "
            "`pip install -r requirements.txt`.",
            file=sys.stderr,
        )
        return 1

    try:
        return solution_main()
    except ValueError as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
