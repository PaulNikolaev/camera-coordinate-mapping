"""Thin CLI wrapper for reusable evaluation logic."""

from __future__ import annotations

import sys


def main() -> int:
    """Import and run the reusable evaluation CLI."""

    try:
        from solution.evaluate import main as solution_main
        from solution.inference import ArtifactLoadError
    except ImportError:
        print(
            "Evaluation dependencies are missing. Install them with "
            "`pip install -r requirements.txt`.",
            file=sys.stderr,
        )
        return 1

    try:
        return solution_main()
    except (ArtifactLoadError, TypeError, ValueError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
