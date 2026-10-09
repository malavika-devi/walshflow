from __future__ import annotations

from pathlib import Path
import unittest

from walshflow.config import load_config
from walshflow.tracking import label_for_offset


ROOT = Path(__file__).resolve().parents[1]


class ConfigAndLabelTests(unittest.TestCase):
    def test_example_configuration(self):
        config = load_config(ROOT / "examples/water/water_angle.toml")
        self.assertEqual(config.system["spin"], 0)
        self.assertEqual(config.scan["type"], "angle")

    def test_wavefunction_method_configurations(self):
        for theory in ("hf", "mp2", "ccsd"):
            config = load_config(ROOT / f"examples/water/water_angle_{theory}.toml")
            self.assertEqual(config.method["theory"], theory)

    def test_orbital_labels(self):
        expected = ["HOMO-2", "HOMO-1", "HOMO", "LUMO", "LUMO+1", "LUMO+2"]
        actual = [label_for_offset(offset) for offset in range(-2, 4)]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
