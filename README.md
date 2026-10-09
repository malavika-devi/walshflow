# WalshFlow

A PySCF workflow for generating structural scans and Walsh diagrams. It runs
each geometry, tracks orbitals between adjacent structures, and writes the
energies and plots to a restartable run directory.

## Install

```bash
mamba env create -f environment.yml
conda activate walshflow
python -m pip install -e .
```

Alternatively:

```bash
python -m pip install -e '.[dispersion]'
```

## Run

The included example scans the H-O-H angle in water:

```bash
walshflow run examples/water/water_angle.toml
```

Preview its geometries without running PySCF:

```bash
walshflow geometries examples/water/water_angle.toml --output /tmp/water_scan
```

Interrupted calculations can be resumed by running the same command again.

## Method settings

Set the electronic-structure method and basis in the `[method]` section of the
TOML configuration.

```toml
# DFT
[method]
theory = "dft"
xc = "pbe0"
basis = "def2-tzvp"
```

```toml
# HF, MP2, or CCSD
[method]
theory = "mp2"          # "hf", "mp2", or "ccsd"
basis = "cc-pvdz"
```

Useful optional settings:

```toml
density_fit = true
conv_tol = 1e-10
max_cycle = 100
memory_mb = 4000
threads = 4
frozen = 0                        # MP2/CCSD only
correlation_conv_tol = 1e-8       # CCSD only
correlation_max_cycle = 100       # CCSD only
```

Element-specific bases and ECPs are accepted:

```toml
basis = { H = "def2-svp", O = "def2-tzvp" }
ecp = { I = "def2-tzvp" }
```

For MP2 and CCSD, the total-energy curve includes the correlation energy. The
orbital curves show the canonical HF reference orbitals.

`system.spin` is `2S`, or multiplicity minus one: `0` for a singlet, `1` for a
doublet, and `2` for a triplet.

## Scan settings

All atom indices are one-based.

```toml
# Bond angle
[scan]
type = "angle"
reference = "structure.xyz"
atoms = [1, 2, 3]
move_atoms = [3]
minimum = 90.0
maximum = 180.0
points = 21
```

Other supported scan types are:

- `bond`: a bond-length scan.
- `interpolate`: linear interpolation between two endpoint structures.
- `cartesian_mode`: displacement along an N-by-3 mode file.
- `normal_mode`: an HF or DFT normal mode calculated with PySCF.

Examples are in [`examples/water`](examples/water). A general normal-mode
template is in [`examples/normal_mode_template.toml`](examples/normal_mode_template.toml).

## Output

Each run directory contains generated XYZ geometries, checkpoints, cached
results, `walsh_data.tsv`, `tracking_diagnostics.json`, and PDF/PNG plots.

Orbitals are matched between adjacent geometries using cross-geometry AO
overlaps and a one-to-one assignment. Inspect `tracking_diagnostics.json` when
assignments are ambiguous.

## HPC

Edit the environment activation in `slurm/run_walsh.slurm`, then submit any
configuration:

```bash
sbatch slurm/run_walsh.slurm examples/water/water_angle.toml
```
