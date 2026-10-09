"""PySCF calculations, restart files, and optional normal-mode generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .config import RunConfig
from .geometry import Geometry, read_geometry


@dataclass
class PointResult:
    index: int
    coordinate: float
    geometry: Geometry
    total_energy: float
    scf_energy: float
    correlation_energy: float
    dispersion_energy: float
    mo_energy: np.ndarray
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    density: np.ndarray | None = None


def _import_pyscf():
    try:
        from pyscf import cc, dft, gto, lib, mp, scf  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PySCF is not installed. Create the environment described in README.md "
            "or run: pip install -e '.[dispersion]'"
        ) from exc
    return dft, gto, lib, scf, mp, cc


def make_molecule(config: RunConfig, geometry: Geometry, *, verbose: int | None = None):
    _, gto, _, _, _, _ = _import_pyscf()
    system = config.system
    method = config.method
    kwargs: dict[str, Any] = {
        "atom": geometry.atom_text(),
        "unit": "Angstrom",
        "charge": int(system["charge"]),
        "spin": int(system["spin"]),
        "basis": method["basis"],
        "symmetry": bool(method.get("symmetry", False)),
        "verbose": int(method.get("pyscf_verbose", 4)) if verbose is None else verbose,
        "max_memory": int(method.get("memory_mb", 4000)),
    }
    if method.get("ecp"):
        kwargs["ecp"] = method["ecp"]
    return gto.M(**kwargs)


def make_mean_field(config: RunConfig, mol, chkfile: Path | None = None):
    _, _, _, _, _, _ = _import_pyscf()
    method = config.method
    theory = str(method.get("theory", "dft")).lower()
    if theory == "dft":
        # Mole.KS chooses RKS for a closed shell and UKS for an open shell.
        mf = mol.KS(xc=str(method["xc"]))
    else:
        # MP2 and CCSD use these HF canonical orbitals as their reference.
        mf = mol.HF()
    if bool(method.get("density_fit", True)):
        auxiliary = method.get("auxbasis")
        mf = mf.density_fit(auxbasis=auxiliary) if auxiliary else mf.density_fit()
    mf.conv_tol = float(method.get("conv_tol", 1e-9))
    mf.max_cycle = int(method.get("max_cycle", 100))
    if hasattr(mf, "grids"):
        mf.grids.level = int(method.get("grid_level", 4))
    if "level_shift" in method:
        mf.level_shift = float(method["level_shift"])
    if "damping" in method:
        mf.damp = float(method["damping"])
    if chkfile is not None:
        mf.chkfile = str(chkfile)
        if chkfile.exists():
            mf.init_guess = "chkfile"
    return mf


def configure_threads(config: RunConfig) -> None:
    _, _, lib, _, _, _ = _import_pyscf()
    lib.num_threads(int(config.method.get("threads", 1)))


def run_point(
    config: RunConfig,
    index: int,
    coordinate: float,
    geometry: Geometry,
    point_dir: Path,
    previous: PointResult | None = None,
) -> PointResult:
    """Run or reload one scan point."""
    point_dir.mkdir(parents=True, exist_ok=True)
    cache = point_dir / "result.npz"
    if cache.exists():
        return load_point(cache, config, geometry)

    mol = make_molecule(config, geometry)
    mf = make_mean_field(config, mol, point_dir / "pyscf.chk")
    initial_density = None
    if previous is not None and previous.density is not None:
        try:
            _, _, _, scf, _, _ = _import_pyscf()
            old_mol = make_molecule(config, previous.geometry)
            initial_density = scf.addons.project_dm_nr2nr(old_mol, previous.density, mol)
        except Exception as exc:
            print(f"  Density projection unavailable; using an atomic guess ({exc})")

    scf_energy = float(mf.kernel(dm0=initial_density))
    if not mf.converged and not bool(config.method.get("accept_unconverged", False)):
        raise RuntimeError(
            f"SCF did not converge for point {index}. Its PySCF checkpoint is in {point_dir}. "
            "Adjust max_cycle/level_shift/damping and rerun."
        )

    correlation_energy = calculate_correlation_energy(config, mf)
    dispersion_energy = calculate_dispersion_energy(config, mol)
    result = PointResult(
        index=index,
        coordinate=float(coordinate),
        geometry=geometry,
        total_energy=scf_energy + correlation_energy + dispersion_energy,
        scf_energy=scf_energy,
        correlation_energy=correlation_energy,
        dispersion_energy=dispersion_energy,
        mo_energy=np.asarray(mf.mo_energy),
        mo_coeff=np.asarray(mf.mo_coeff),
        mo_occ=np.asarray(mf.mo_occ),
        density=np.asarray(mf.make_rdm1()),
    )
    save_point(cache, config, result)
    (point_dir / "geometry.xyz").write_text(
        geometry.xyz_text(f"point={index} coordinate={coordinate:.12g}")
    )
    return result


def save_point(path: Path, config: RunConfig, result: PointResult) -> None:
    metadata = json.dumps({"config_digest": config.digest, "index": result.index})
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            metadata=np.asarray(metadata),
            coordinate=np.asarray(result.coordinate),
            symbols=np.asarray(result.geometry.symbols),
            coordinates=result.geometry.coordinates,
            total_energy=np.asarray(result.total_energy),
            scf_energy=np.asarray(result.scf_energy),
            correlation_energy=np.asarray(result.correlation_energy),
            dispersion_energy=np.asarray(result.dispersion_energy),
            mo_energy=result.mo_energy,
            mo_coeff=result.mo_coeff,
            mo_occ=result.mo_occ,
            density=result.density if result.density is not None else np.asarray([]),
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_point(path: Path, config: RunConfig, expected_geometry: Geometry) -> PointResult:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata["config_digest"] != config.digest:
            raise RuntimeError(
                f"Cached result {path} was produced from a different configuration. "
                "Use a new output.directory or remove that run directory."
            )
        symbols = tuple(str(value) for value in data["symbols"])
        geometry = Geometry(symbols, data["coordinates"])
        if symbols != expected_geometry.symbols or not np.allclose(
            geometry.coordinates, expected_geometry.coordinates, atol=1e-10
        ):
            raise RuntimeError(f"Cached geometry does not match the requested scan at {path}")
        density = np.asarray(data["density"])
        return PointResult(
            index=int(metadata["index"]),
            coordinate=float(data["coordinate"]),
            geometry=geometry,
            total_energy=float(data["total_energy"]),
            scf_energy=float(data["scf_energy"]) if "scf_energy" in data else float(data["total_energy"]),
            correlation_energy=float(data["correlation_energy"]) if "correlation_energy" in data else 0.0,
            dispersion_energy=float(data["dispersion_energy"]) if "dispersion_energy" in data else 0.0,
            mo_energy=np.asarray(data["mo_energy"]),
            mo_coeff=np.asarray(data["mo_coeff"]),
            mo_occ=np.asarray(data["mo_occ"]),
            density=density if density.size else None,
        )


def calculate_correlation_energy(config: RunConfig, mf) -> float:
    theory = str(config.method.get("theory", "dft")).lower()
    if theory in {"dft", "hf"}:
        return 0.0
    _, _, _, _, mp, cc = _import_pyscf()
    if theory == "mp2":
        solver = mp.MP2(mf)
    elif theory == "ccsd":
        solver = cc.CCSD(mf)
        solver.max_cycle = int(config.method.get("correlation_max_cycle", 100))
        solver.conv_tol = float(config.method.get("correlation_conv_tol", 1e-7))
    else:
        raise ValueError(f"Unsupported correlated method {theory!r}")
    if "frozen" in config.method:
        solver.frozen = config.method["frozen"]
    solver.max_memory = int(config.method.get("memory_mb", 4000))
    solver.kernel()
    if hasattr(solver, "converged") and not solver.converged:
        raise RuntimeError(f"{theory.upper()} did not converge")
    return float(solver.e_corr)


def _dispersion_objects(config: RunConfig, mol):
    version = str(config.method.get("dispersion", "")).strip().lower()
    if not version:
        return None, None
    try:
        from dftd3.interface import (  # type: ignore
            DispersionModel,
            RationalDampingParam,
            ZeroDampingParam,
        )
        from pyscf import gto  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "method.dispersion requires the maintained simple-dftd3 Python package. "
            "Install the 'dispersion' extra or run: pip install dftd3"
        ) from exc
    parameter_classes = {"d3bj": RationalDampingParam, "d3zero": ZeroDampingParam}
    if version not in parameter_classes:
        raise ValueError("method.dispersion currently supports 'd3bj' and 'd3zero'")
    numbers = np.asarray(
        [gto.charge(mol.atom_pure_symbol(index)) for index in range(mol.natm)], dtype=int
    )
    # Both PySCF and the simple-dftd3 API use Bohr here.
    model = DispersionModel(numbers=numbers, positions=np.asarray(mol.atom_coords()))
    parameter_method = str(
        config.method.get(
            "dispersion_parameter_method",
            config.method.get("xc", config.method.get("theory", "hf")),
        )
    ).lower()
    parameter = parameter_classes[version](
        method=parameter_method,
        atm=bool(config.method.get("dispersion_three_body", False)),
    )
    return model, parameter


def calculate_dispersion_energy(config: RunConfig, mol) -> float:
    model, parameter = _dispersion_objects(config, mol)
    if model is None:
        return 0.0
    return float(model.get_dispersion(parameter, grad=False)["energy"])


def calculate_dispersion_hessian(config: RunConfig, mol) -> np.ndarray | None:
    model, parameter = _dispersion_objects(config, mol)
    if model is None:
        return None
    return np.asarray(model.get_hessian(parameter)["hessian"], dtype=float)


def build_normal_mode_scan(config: RunConfig, work_dir: Path) -> tuple[np.ndarray, list[Geometry]]:
    """Calculate a reference Hessian and displace along a selected normal mode.

    The selected PySCF normal mode is Euclidean-normalized. Consequently the
    configured minimum/maximum values are total Cartesian displacement norms in
    Angstrom, making the amplitude definition explicit and reproducible.
    """
    try:
        from pyscf.hessian import thermo  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PySCF Hessian support is unavailable") from exc

    scan = config.scan
    reference = read_geometry(config.resolve(scan["reference"]))
    normal_dir = work_dir / "normal_mode_reference"
    normal_dir.mkdir(parents=True, exist_ok=True)
    saved_mode = normal_dir / "selected_mode.txt"
    saved_summary = normal_dir / "mode_summary.json"
    values = np.linspace(float(scan["minimum"]), float(scan["maximum"]), int(scan["points"]))
    if saved_mode.exists() and saved_summary.exists():
        summary = json.loads(saved_summary.read_text())
        if summary.get("config_digest") != config.digest:
            raise RuntimeError(
                "The cached normal mode belongs to a different configuration; "
                "select a new output.directory or remove the old run directory."
            )
        mode = np.loadtxt(saved_mode).reshape(reference.coordinates.shape)
        return values, [
            Geometry(reference.symbols, reference.coordinates + value * mode) for value in values
        ]

    mol = make_molecule(config, reference)
    mf = make_mean_field(config, mol, normal_dir / "pyscf.chk")
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError("Reference SCF did not converge; cannot calculate its Hessian")
    hessian = mf.Hessian().kernel()
    dispersion_hessian = calculate_dispersion_hessian(config, mol)
    if dispersion_hessian is not None:
        natom = mol.natm
        electronic_matrix = hessian.transpose(0, 2, 1, 3).reshape(3 * natom, 3 * natom)
        hessian = (electronic_matrix + dispersion_hessian).reshape(
            natom, 3, natom, 3
        ).transpose(0, 2, 1, 3)
    analysis = thermo.harmonic_analysis(mol, hessian, imaginary_freq=True)
    frequencies = np.asarray(analysis["freq_wavenumber"])
    modes = np.asarray(analysis["norm_mode"])
    signed = np.asarray(
        [-abs(value.imag) if abs(value.imag) > 1e-10 else value.real for value in frequencies]
    )
    order = np.argsort(signed)
    rank = int(scan.get("mode_index", 0))
    if rank < 0 or rank >= len(order):
        raise ValueError(f"normal-mode mode_index {rank} is outside 0..{len(order)-1}")
    selected = int(order[rank])
    mode = np.real_if_close(modes[selected]).astype(float)
    if mode.shape != reference.coordinates.shape:
        mode = mode.reshape(reference.coordinates.shape)
    mode /= np.linalg.norm(mode)

    geometries = [
        Geometry(reference.symbols, reference.coordinates + value * mode) for value in values
    ]
    np.savetxt(saved_mode, mode, fmt="% .12e")
    summary = {
        "config_digest": config.digest,
        "electronic_energy_hartree": float(energy),
        "dispersion_energy_hartree": calculate_dispersion_energy(config, mol),
        "selected_mode": selected,
        "selection_rank": rank,
        "frequency_cm-1": float(signed[selected]),
        "amplitude_unit": "Angstrom (Euclidean-normalized Cartesian mode)",
    }
    saved_summary.write_text(json.dumps(summary, indent=2) + "\n")
    return values, geometries
