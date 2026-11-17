"""Acceptance runner: unavailable FMM checks are incomplete and fail full acceptance."""

import argparse
from pathlib import Path
import sys
import unittest


def main():
    """Run the Python correctness suite and enforce full versus direct-only acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-only", action="store_true",
                        help="allow incomplete FMM checks; does not claim full acceptance")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print(f"INCOMPLETE: {len(result.skipped)} FMM-dependent tests were skipped", file=sys.stderr)
    return 0 if result.wasSuccessful() and (not result.skipped or args.direct_only) else 1


if __name__ == "__main__":
    sys.exit(main())
