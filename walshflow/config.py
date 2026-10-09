"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tomllib
from typing import Any


class ConfigError(ValueError):
    """Raised when the TOML configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class RunConfig:
    path: Path
    base_dir: Path
    raw: dict[str, Any]
    digest: str

    @property
    def system(self) -> dict[str, Any]:
        return self.raw["system"]

    @property
    def method(self) -> dict[str, Any]:
        return self.raw["method"]

    @property
    def scan(self) -> dict[str, Any]:
        return self.raw["scan"]

    @property
    def tracking(self) -> dict[str, Any]:
        return self.raw.get("tracking", {})

    @property
    def output(self) -> dict[str, Any]:
        return self.raw.get("output", {})

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.base_dir / path).resolve()


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path).expanduser().resolve()
    payload = config_path.read_bytes()
    raw = tomllib.loads(payload.decode("utf-8"))

    for section in ("system", "method", "scan"):
        if section not in raw or not isinstance(raw[section], dict):
            raise ConfigError(f"Missing [{section}] section in {config_path}")

    system = raw["system"]
    method = raw["method"]
    scan = raw["scan"]
    for key in ("name", "charge", "spin"):
        if key not in system:
            raise ConfigError(f"[system] requires {key!r}")
    if "basis" not in method:
        raise ConfigError("[method] requires 'basis'")
    theory = str(method.get("theory", "dft" if "xc" in method else "hf")).lower()
    supported_theories = {"dft", "hf", "mp2", "ccsd"}
    if theory not in supported_theories:
        raise ConfigError(
            f"Unsupported method.theory {theory!r}; choose {', '.join(sorted(supported_theories))}"
        )
    method["theory"] = theory
    if theory == "dft" and "xc" not in method:
        raise ConfigError("[method] requires 'xc' when theory = 'dft'")
    if theory in {"mp2", "ccsd"} and method.get("dispersion"):
        raise ConfigError(
            "Automatic D3 dispersion is not enabled for MP2/CCSD. Remove method.dispersion "
            "or use a separately justified correlated composite method."
        )
    if "type" not in scan:
        raise ConfigError("[scan] requires 'type'")

    scan_type = str(scan["type"]).lower()
    required = {
        "interpolate": ("start", "end", "points"),
        "cartesian_mode": ("reference", "mode_file", "minimum", "maximum", "points"),
        "bond": ("reference", "atoms", "minimum", "maximum", "points"),
        "angle": ("reference", "atoms", "minimum", "maximum", "points"),
        "normal_mode": ("reference", "minimum", "maximum", "points"),
    }
    if scan_type not in required:
        raise ConfigError(
            f"Unsupported scan.type {scan_type!r}; choose {', '.join(required)}"
        )
    missing = [key for key in required[scan_type] if key not in scan]
    if missing:
        raise ConfigError(f"[scan] type {scan_type!r} requires: {', '.join(missing)}")

    if int(scan.get("points", 0)) < 2:
        raise ConfigError("scan.points must be at least 2")
    if scan_type == "normal_mode" and theory in {"mp2", "ccsd"}:
        raise ConfigError(
            "normal_mode currently supports DFT and HF Hessians. For MP2/CCSD energies, "
            "use cartesian_mode or an endpoint/coordinate scan."
        )
    if int(system["spin"]) < 0:
        raise ConfigError("system.spin is 2S (multiplicity - 1) and cannot be negative")

    digest = hashlib.sha256(payload).hexdigest()
    return RunConfig(config_path, config_path.parent, raw, digest)
