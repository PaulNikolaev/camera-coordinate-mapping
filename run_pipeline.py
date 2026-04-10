"""Thin CLI wrapper for reusable train-and-evaluate pipeline logic."""

from __future__ import annotations

import sys


def main() -> int:
    """Import and run the reusable pipeline CLI."""
    try:
        from solution.inference import ArtifactLoadError
        from solution.pipeline import main as solution_main
    except ImportError:
        print(
            "Pipeline dependencies are missing. Install them with "
            "`pip install -r requirements.txt`.",
            file=sys.stderr,
        )
        return 1

    try:
        return solution_main()
    except (ArtifactLoadError, TypeError, ValueError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
