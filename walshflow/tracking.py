"""Cross-geometry AO-overlap orbital tracking."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import RunConfig
from .qc import PointResult, make_molecule


HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class OrbitalMatch:
    energy_ev: float
    raw_index: int  # zero based internally
    overlap: float
    margin: float


@dataclass
class TrackedResult:
    points: list[PointResult]
    labels: list[str]
    matches: list[dict[str, OrbitalMatch]]
    reference_index: int


def label_for_offset(offset: int) -> str:
    if offset < 0:
        return f"HOMO{offset}"
    if offset == 0:
        return "HOMO"
    if offset == 1:
        return "LUMO"
    return f"LUMO+{offset - 1}"


def _channel(point: PointResult, channel: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if point.mo_energy.ndim == 1:
        return point.mo_energy, point.mo_coeff, point.mo_occ
    index = 0 if channel.lower() == "alpha" else 1
    return point.mo_energy[index], point.mo_coeff[index], point.mo_occ[index]


def _homo_index(occupations: np.ndarray) -> int:
    occupied = np.flatnonzero(np.asarray(occupations) > 1e-8)
    if not len(occupied):
        raise ValueError("No occupied orbitals were found")
    return int(occupied[-1])


def track_orbitals(config: RunConfig, points: list[PointResult]) -> TrackedResult:
    tracking = config.tracking
    n_orbitals = int(tracking.get("orbitals_each_side", 2))
    margin = int(tracking.get("candidate_margin", 4))
    max_delta_ev = float(tracking.get("max_energy_step_ev", 1.5))
    channel = str(tracking.get("spin_channel", "alpha"))
    if n_orbitals < 0 or margin < 0:
        raise ValueError("tracking orbital counts and margin cannot be negative")

    offsets = list(range(-n_orbitals, n_orbitals + 2))
    labels = [label_for_offset(offset) for offset in offsets]
    requested_reference = tracking.get("reference_index")
    reference = (
        int(requested_reference)
        if requested_reference is not None
        else int(np.argmin(np.abs([point.coordinate for point in points])))
    )
    if reference < 0 or reference >= len(points):
        raise ValueError("tracking.reference_index is outside the scan")

    ref_energy, _, ref_occ = _channel(points[reference], channel)
    ref_homo = _homo_index(ref_occ)
    ref_indices = [ref_homo + offset for offset in offsets]
    if min(ref_indices) < 0 or max(ref_indices) >= len(ref_energy):
        raise ValueError("Requested reference orbital window lies outside available MOs")

    matches: list[dict[str, OrbitalMatch]] = [dict() for _ in points]
    for label, raw_index in zip(labels, ref_indices):
        matches[reference][label] = OrbitalMatch(
            float(ref_energy[raw_index] * HARTREE_TO_EV), raw_index, np.nan, np.nan
        )

    for direction in (1, -1):
        previous_index = reference
        target_index = reference + direction
        while 0 <= target_index < len(points):
            matches[target_index] = _match_adjacent(
                config,
                points[previous_index],
                points[target_index],
                matches[previous_index],
                labels,
                n_orbitals,
                margin,
                max_delta_ev,
                channel,
            )
            previous_index = target_index
            target_index += direction

    return TrackedResult(points, labels, matches, reference)


def _match_adjacent(
    config: RunConfig,
    previous: PointResult,
    target: PointResult,
    previous_matches: dict[str, OrbitalMatch],
    labels: list[str],
    n_orbitals: int,
    margin: int,
    max_delta_ev: float,
    channel: str,
) -> dict[str, OrbitalMatch]:
    try:
        from pyscf import gto  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PySCF is required to evaluate cross-geometry AO overlaps") from exc

    previous_energy, previous_coeff, _ = _channel(previous, channel)
    target_energy, target_coeff, target_occ = _channel(target, channel)
    target_homo = _homo_index(target_occ)
    low = max(0, target_homo - n_orbitals - margin)
    high = min(len(target_energy), target_homo + n_orbitals + margin + 2)
    candidates = np.arange(low, high, dtype=int)
    previous_indices = np.asarray([previous_matches[label].raw_index for label in labels])

    previous_mol = make_molecule(config, previous.geometry, verbose=0)
    target_mol = make_molecule(config, target.geometry, verbose=0)
    ao_cross = gto.intor_cross("int1e_ovlp", previous_mol, target_mol)
    overlap = np.abs(
        previous_coeff[:, previous_indices].T @ ao_cross @ target_coeff[:, candidates]
    )
    previous_ev = previous_energy[previous_indices] * HARTREE_TO_EV
    candidate_ev = target_energy[candidates] * HARTREE_TO_EV
    allowed = np.abs(previous_ev[:, None] - candidate_ev[None, :]) <= max_delta_ev
    for row, label in enumerate(labels):
        if not np.any(allowed[row]):
            nearest = float(np.min(np.abs(previous_ev[row] - candidate_ev)))
            raise RuntimeError(
                f"No allowed match for {label} from point {previous.index} to {target.index}; "
                f"nearest energy change is {nearest:.3f} eV but max_energy_step_ev is "
                f"{max_delta_ev:.3f}. Increase the threshold or use more scan points."
            )

    score = np.where(allowed, overlap, -1.0e9)
    rows, columns = linear_sum_assignment(score, maximize=True)
    assignment = dict(zip(rows.tolist(), columns.tolist()))
    if len(assignment) != len(labels):
        raise RuntimeError("The candidate window is too small for one-to-one orbital assignment")

    result: dict[str, OrbitalMatch] = {}
    for row, label in enumerate(labels):
        column = assignment[row]
        if not allowed[row, column]:
            raise RuntimeError(
                f"One-to-one matching forced a disallowed assignment for {label}; "
                "increase candidate_margin or max_energy_step_ev."
            )
        alternatives = score[row].copy()
        alternatives[column] = -1.0e9
        finite_alternatives = alternatives[alternatives > -1.0e8]
        runner_up = float(np.max(finite_alternatives)) if finite_alternatives.size else 0.0
        raw = int(candidates[column])
        result[label] = OrbitalMatch(
            energy_ev=float(target_energy[raw] * HARTREE_TO_EV),
            raw_index=raw,
            overlap=float(overlap[row, column]),
            margin=float(overlap[row, column] - runner_up),
        )
    return result
