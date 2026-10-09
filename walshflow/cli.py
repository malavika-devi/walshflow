"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .config import load_config
from .geometry import build_scan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="walshflow",
        description="Run a restartable PySCF scan and make an identity-tracked Walsh diagram.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="Run/resume calculations, track MOs, and plot")
    run.add_argument("config", type=Path, help="TOML configuration file")

    geometries = subcommands.add_parser(
        "geometries", help="Generate scan XYZ files without running PySCF"
    )
    geometries.add_argument("config", type=Path, help="TOML configuration file")
    geometries.add_argument("--output", type=Path, default=Path("generated_geometries"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "run":
        from .pipeline import run_workflow

        run_workflow(config)
        return

    if str(config.scan["type"]).lower() == "normal_mode":
        raise SystemExit(
            "A normal_mode scan requires a PySCF Hessian. Use 'walshflow run'; "
            "the other scan types can be previewed without calculations."
        )
    coordinates, scan = build_scan(config)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for index, (coordinate, geometry) in enumerate(zip(coordinates, scan)):
        (output / f"point_{index:03d}.xyz").write_text(
            geometry.xyz_text(f"coordinate={coordinate:.12g}")
        )
    print(f"Wrote {len(scan)} geometries to {output}")


if __name__ == "__main__":
    main()
