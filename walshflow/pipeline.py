"""End-to-end Walsh workflow orchestration and output generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import RunConfig
from .geometry import build_scan
from .qc import PointResult, build_normal_mode_scan, configure_threads, run_point
from .tracking import TrackedResult, track_orbitals


HARTREE_TO_KCAL_MOL = 627.5094740631
COLORS = [
    "#6A3D9A", "#008080", "#0B2545", "#C2185B",
    "#E66100", "#1F78B4", "#2E8B57", "#8B4513",
]


def run_workflow(config: RunConfig) -> Path:
    output_value = config.output.get("directory", "run")
    work_dir = config.resolve(output_value)
    work_dir.mkdir(parents=True, exist_ok=True)
    _prepare_manifest(config, work_dir)
    configure_threads(config)

    if str(config.scan["type"]).lower() == "normal_mode":
        coordinates, geometries = build_normal_mode_scan(config, work_dir)
    else:
        coordinates, geometries = build_scan(config)

    geometry_dir = work_dir / "geometries"
    geometry_dir.mkdir(exist_ok=True)
    for index, (coordinate, geometry) in enumerate(zip(coordinates, geometries)):
        (geometry_dir / f"point_{index:03d}.xyz").write_text(
            geometry.xyz_text(f"coordinate={coordinate:.12g}")
        )

    points: list[PointResult] = []
    previous: PointResult | None = None
    print(f"Running {len(geometries)} scan points in {work_dir}")
    for index, (coordinate, geometry) in enumerate(zip(coordinates, geometries)):
        point_dir = work_dir / "points" / f"point_{index:03d}"
        status = "reload" if (point_dir / "result.npz").exists() else "calculate"
        print(f"[{index + 1:>3d}/{len(geometries)}] {status:>9s} coordinate={coordinate:.8g}")
        point = run_point(config, index, float(coordinate), geometry, point_dir, previous)
        points.append(point)
        previous = point

    print("Tracking orbitals with cross-geometry AO overlaps")
    tracked = track_orbitals(config, points)
    tsv_path = work_dir / "walsh_data.tsv"
    _write_tsv(tsv_path, tracked)
    _write_diagnostics(work_dir / "tracking_diagnostics.json", tracked, config)
    _plot(work_dir / "walsh_diagram.pdf", tracked, config)
    _plot(work_dir / "walsh_diagram.png", tracked, config)
    print(f"Walsh data    -> {tsv_path}")
    print(f"Walsh diagram -> {work_dir / 'walsh_diagram.pdf'}")
    return work_dir


def _prepare_manifest(config: RunConfig, work_dir: Path) -> None:
    manifest_path = work_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config_digest") != config.digest:
            raise RuntimeError(
                f"{work_dir} contains a run from a different configuration. "
                "Choose a new output.directory to preserve the previous calculation."
            )
        return
    shutil.copy2(config.path, work_dir / "config.toml")
    manifest_path.write_text(
        json.dumps(
            {"config_digest": config.digest, "source_config": str(config.path)}, indent=2
        ) + "\n"
    )


def _energy_reference(tracked: TrackedResult, config: RunConfig) -> float:
    energies = np.asarray([point.total_energy for point in tracked.points])
    choice = str(config.output.get("energy_reference", "reference")).lower()
    if choice == "minimum":
        return float(np.min(energies))
    if choice == "first":
        return float(energies[0])
    if choice == "reference":
        return float(energies[tracked.reference_index])
    raise ValueError("output.energy_reference must be reference, minimum, or first")


def _write_tsv(path: Path, tracked: TrackedResult) -> None:
    fields = [
        "step", "coordinate", "reference_energy_hartree", "correlation_energy_hartree",
        "dispersion_energy_hartree",
        "total_energy_hartree",
    ]
    for label in tracked.labels:
        fields.extend(
            [f"{label}_eV", f"{label}_rawMO", f"{label}_overlap", f"{label}_margin"]
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, point in enumerate(tracked.points):
            row: dict[str, str | int] = {
                "step": index,
                "coordinate": f"{point.coordinate:.10g}",
                "reference_energy_hartree": f"{point.scf_energy:.12f}",
                "correlation_energy_hartree": f"{point.correlation_energy:.12f}",
                "dispersion_energy_hartree": f"{point.dispersion_energy:.12f}",
                "total_energy_hartree": f"{point.total_energy:.12f}",
            }
            for label in tracked.labels:
                match = tracked.matches[index][label]
                row[f"{label}_eV"] = f"{match.energy_ev:.8f}"
                row[f"{label}_rawMO"] = match.raw_index + 1  # human-facing 1-based MO
                row[f"{label}_overlap"] = "nan" if np.isnan(match.overlap) else f"{match.overlap:.8f}"
                row[f"{label}_margin"] = "nan" if np.isnan(match.margin) else f"{match.margin:.8f}"
            writer.writerow(row)


def _write_diagnostics(path: Path, tracked: TrackedResult, config: RunConfig) -> None:
    threshold = float(config.tracking.get("ambiguous_margin", 0.10))
    warnings = []
    for step, matches in enumerate(tracked.matches):
        for label, match in matches.items():
            if not np.isnan(match.margin) and match.margin < threshold:
                warnings.append(
                    {
                        "step": step,
                        "label": label,
                        "raw_mo": match.raw_index + 1,
                        "overlap": match.overlap,
                        "margin": match.margin,
                    }
                )
    payload = {
        "reference_index": tracked.reference_index,
        "ambiguous_margin_threshold": threshold,
        "ambiguous_matches": warnings,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    if warnings:
        print(f"WARNING: {len(warnings)} low-margin orbital assignments; inspect {path}")


def _plot(path: Path, tracked: TrackedResult, config: RunConfig) -> None:
    x = np.asarray([point.coordinate for point in tracked.points])
    reference_energy = _energy_reference(tracked, config)
    relative = np.asarray(
        [(point.total_energy - reference_energy) * HARTREE_TO_KCAL_MOL for point in tracked.points]
    )

    fig, axis = plt.subplots(figsize=(9.2, 6.2))
    for index, label in enumerate(tracked.labels):
        energy = np.asarray([matches[label].energy_ev for matches in tracked.matches])
        frontier = label in {"HOMO", "LUMO"}
        axis.plot(
            x,
            energy,
            color=COLORS[index % len(COLORS)],
            linewidth=3.0 if frontier else 2.0,
            linestyle="-" if frontier else "--",
            label=label,
        )
    axis.set_xlabel(str(config.output.get("coordinate_label", "Reaction coordinate")))
    axis.set_ylabel("MO energy (eV)")
    axis.grid(axis="y", linestyle="--", alpha=0.45)

    energy_axis = axis.twinx()
    energy_axis.plot(x, relative, color="#111111", linewidth=2.5, label="Total energy")
    energy_axis.set_ylabel("Relative total energy (kcal mol$^{-1}$)")

    handles, labels = axis.get_legend_handles_labels()
    energy_handles, energy_labels = energy_axis.get_legend_handles_labels()
    axis.legend(handles + energy_handles, labels + energy_labels, ncol=2, fontsize=9)
    title = str(config.output.get("title", f"{config.system['name']} Walsh diagram"))
    axis.set_title(title)
    fig.tight_layout()
    if path.suffix.lower() == ".png":
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
