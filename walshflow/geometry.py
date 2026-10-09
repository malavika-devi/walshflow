"""Geometry readers and deterministic scan-coordinate builders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from .config import ConfigError, RunConfig


@dataclass(frozen=True)
class Geometry:
    symbols: tuple[str, ...]
    coordinates: np.ndarray  # Angstrom, shape (natom, 3)

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates, dtype=float)
        if coords.shape != (len(self.symbols), 3):
            raise ValueError("coordinates must have shape (number of atoms, 3)")
        object.__setattr__(self, "coordinates", coords)

    def atom_text(self) -> str:
        return "\n".join(
            f"{symbol:<3s} {x: .12f} {y: .12f} {z: .12f}"
            for symbol, (x, y, z) in zip(self.symbols, self.coordinates)
        )

    def xyz_text(self, comment: str = "") -> str:
        return f"{len(self.symbols)}\n{comment}\n{self.atom_text()}\n"


def read_geometry(path: str | Path) -> Geometry:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xyz":
        return _read_xyz(path)
    if suffix in {".gjf", ".com"}:
        return _read_gaussian_input(path)
    raise ValueError(f"Unsupported geometry format {suffix!r}: use .xyz, .gjf, or .com")


def _parse_atom_lines(lines: Iterable[str]) -> Geometry:
    symbols: list[str] = []
    coords: list[list[float]] = []
    atom_pattern = re.compile(r"^\s*([A-Za-z]{1,3})\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)")
    for line in lines:
        match = atom_pattern.match(line)
        if not match:
            continue
        symbols.append(match.group(1))
        coords.append([float(value.replace("D", "E").replace("d", "e")) for value in match.groups()[1:]])
    if not symbols:
        raise ValueError("No Cartesian atom records found")
    return Geometry(tuple(symbols), np.asarray(coords))


def _read_xyz(path: Path) -> Geometry:
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty XYZ file: {path}")
    try:
        natom = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"First line of {path} must contain the atom count") from exc
    geometry = _parse_atom_lines(lines[2 : 2 + natom])
    if len(geometry.symbols) != natom:
        raise ValueError(f"Expected {natom} atoms in {path}, found {len(geometry.symbols)}")
    return geometry


def _read_gaussian_input(path: Path) -> Geometry:
    lines = path.read_text(errors="replace").splitlines()
    charge_line = None
    charge_pattern = re.compile(r"^\s*[+-]?\d+\s+\d+\s*$")
    for index, line in enumerate(lines):
        if charge_pattern.match(line):
            charge_line = index
            break
    if charge_line is None:
        raise ValueError(f"Could not find charge/multiplicity line in {path}")
    atom_lines: list[str] = []
    for line in lines[charge_line + 1 :]:
        if not line.strip():
            break
        atom_lines.append(line)
    return _parse_atom_lines(atom_lines)


def _same_atoms(first: Geometry, second: Geometry) -> None:
    if first.symbols != second.symbols:
        raise ValueError("Scan endpoint geometries have different atom symbols or ordering")


def _values(scan: dict) -> np.ndarray:
    return np.linspace(float(scan["minimum"]), float(scan["maximum"]), int(scan["points"]))


def build_scan(config: RunConfig) -> tuple[np.ndarray, list[Geometry]]:
    """Build all non-Hessian scan types from the configuration."""
    scan = config.scan
    scan_type = str(scan["type"]).lower()
    if scan_type == "normal_mode":
        raise ConfigError("normal_mode geometry generation is handled by the PySCF pipeline")

    if scan_type == "interpolate":
        start = read_geometry(config.resolve(scan["start"]))
        end = read_geometry(config.resolve(scan["end"]))
        _same_atoms(start, end)
        points = int(scan["points"])
        fractions = np.linspace(0.0, 1.0, points)
        coordinate = np.linspace(
            float(scan.get("minimum", 0.0)), float(scan.get("maximum", 1.0)), points
        )
        geometries = [
            Geometry(start.symbols, (1.0 - fraction) * start.coordinates + fraction * end.coordinates)
            for fraction in fractions
        ]
        return coordinate, geometries

    reference = read_geometry(config.resolve(scan["reference"]))
    values = _values(scan)

    if scan_type == "cartesian_mode":
        mode = np.loadtxt(config.resolve(scan["mode_file"]), dtype=float)
        if mode.shape != reference.coordinates.shape:
            raise ValueError(
                f"Mode shape {mode.shape} does not match geometry shape {reference.coordinates.shape}"
            )
        norm = float(np.linalg.norm(mode))
        if norm == 0:
            raise ValueError("Cartesian mode vector has zero norm")
        if bool(scan.get("normalize_mode", True)):
            mode = mode / norm
        return values, [Geometry(reference.symbols, reference.coordinates + value * mode) for value in values]

    atoms = [int(index) - 1 for index in scan["atoms"]]
    if any(index < 0 or index >= len(reference.symbols) for index in atoms):
        raise ValueError("scan.atoms uses 1-based atom numbers and contains an out-of-range value")

    if scan_type == "bond":
        if len(atoms) != 2:
            raise ValueError("A bond scan requires exactly two atom numbers")
        fixed, moving_anchor = atoms
        moving = _moving_atoms(scan, moving_anchor, len(reference.symbols))
        vector = reference.coordinates[moving_anchor] - reference.coordinates[fixed]
        distance = float(np.linalg.norm(vector))
        if distance == 0:
            raise ValueError("Bond-defining atoms occupy the same point")
        unit = vector / distance
        geometries = []
        for target in values:
            coords = reference.coordinates.copy()
            coords[moving] += (target - distance) * unit
            geometries.append(Geometry(reference.symbols, coords))
        return values, geometries

    if scan_type == "angle":
        if len(atoms) != 3:
            raise ValueError("An angle scan requires atom numbers [outer, vertex, outer]")
        first, vertex, moving_anchor = atoms
        moving = _moving_atoms(scan, moving_anchor, len(reference.symbols))
        v1 = reference.coordinates[first] - reference.coordinates[vertex]
        v2 = reference.coordinates[moving_anchor] - reference.coordinates[vertex]
        current = _angle_degrees(v1, v2)
        axis = np.cross(v1, v2)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            raise ValueError("Cannot define an angle-rotation plane from collinear atoms")
        axis /= axis_norm
        geometries = []
        center = reference.coordinates[vertex]
        for target in values:
            rotation = _rotation_matrix(axis, np.radians(target - current))
            coords = reference.coordinates.copy()
            coords[moving] = (coords[moving] - center) @ rotation.T + center
            geometries.append(Geometry(reference.symbols, coords))
        return values, geometries

    raise AssertionError(f"Unhandled scan type {scan_type}")


def _moving_atoms(scan: dict, anchor: int, natom: int) -> np.ndarray:
    raw = scan.get("move_atoms", [anchor + 1])
    moving = np.asarray([int(index) - 1 for index in raw], dtype=int)
    if moving.size == 0 or np.any(moving < 0) or np.any(moving >= natom):
        raise ValueError("scan.move_atoms must contain valid 1-based atom numbers")
    if anchor not in moving:
        raise ValueError("scan.move_atoms must include the moving outer/anchor atom")
    return moving


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _rotation_matrix(axis: np.ndarray, radians: float) -> np.ndarray:
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    identity = np.eye(3)
    return identity * np.cos(radians) + (1.0 - np.cos(radians)) * np.outer(axis, axis) + np.sin(radians) * cross
