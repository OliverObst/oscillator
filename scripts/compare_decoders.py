"""Fit the fixed three-seed harmonic/RFF comparison, then evaluate the frozen choices."""

import argparse
from pathlib import Path

from oscillator.experiment import compare_decoders, evaluate_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("runs/comparison"))
    parser.add_argument("--skeleton", type=Path, default=Path("web/assets/skeleton.json"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--evaluate-selection", action="store_true", help="Resume a frozen selection"
    )
    args = parser.parse_args()
    if args.evaluate_selection:
        evaluate_selection(args.data, args.output, args.skeleton)
    else:
        compare_decoders(args.data, args.output, args.skeleton, args.epochs)


if __name__ == "__main__":
    main()
