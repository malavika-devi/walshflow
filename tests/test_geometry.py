from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from walshflow.config import load_config
from walshflow.geometry import build_scan, read_geometry


ROOT = Path(__file__).resolve().parents[1]


def angle_degrees(a, b, c):
    first = a - b
    second = c - b
    cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


class GeometryTests(unittest.TestCase):
    def test_water_angle_scan_reaches_requested_limits(self):
        config = load_config(ROOT / "examples/water/water_angle.toml")
        values, geometries = build_scan(config)
        self.assertEqual(len(geometries), 19)
        self.assertAlmostEqual(values[0], 90.0)
        self.assertAlmostEqual(values[-1], 180.0)
        for expected, geometry in zip((values[0], values[-1]), (geometries[0], geometries[-1])):
            actual = angle_degrees(
                geometry.coordinates[1], geometry.coordinates[0], geometry.coordinates[2]
            )
            self.assertAlmostEqual(actual, expected, places=7)

    def test_gaussian_cartesian_reader(self):
        text = """# test\n\nTitle\n\n0 1\nH 0 0 0\nH 0 0 0.74\n\n"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h2.gjf"
            path.write_text(text)
            geometry = read_geometry(path)
        self.assertEqual(geometry.symbols, ("H", "H"))
        self.assertAlmostEqual(geometry.coordinates[1, 2], 0.74)

if __name__ == "__main__":
    unittest.main()
