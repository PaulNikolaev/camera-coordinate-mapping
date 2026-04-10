"""Thin CLI wrapper for reusable coordinate prediction logic."""

from __future__ import annotations

import sys


def main() -> int:
    """Import and run the reusable prediction CLI."""

    try:
        from solution.inference import ArtifactLoadError, main as solution_main
    except ImportError:
        print(
            "Prediction dependencies are missing. Install them with "
            "`pip install -r requirements.txt`.",
            file=sys.stderr,
        )
        return 1

    try:
        return solution_main()
    except (ArtifactLoadError, TypeError, ValueError) as error:
        print(f"Prediction failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
