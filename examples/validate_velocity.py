"""Run Python normalization, field, wall and reciprocity checks and optionally save JSON."""

import argparse
import json
import math
from pathlib import Path

from blake.validation import provenance, validate_velocity, validate_wrappers


def clean_json(value):
    """Encode undefined relative wall errors as JSON null instead of nonstandard NaN."""
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    """Collect checked numerical measurements, dependency paths and software versions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON report destination")
    args = parser.parse_args()
    report = dict(provenance=provenance(), normalization=validate_wrappers(), **validate_velocity())
    text = json.dumps(clean_json(report), indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"Validated {len(report['cases'])} field cases and 3 reciprocity tolerances: {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
