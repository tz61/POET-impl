"""Checkpoint task compatibility tests; no simulator is imported or mocked."""
import json
from pathlib import Path
import tempfile
import unittest

from isaaclab_port.task_fingerprint import (
    build_task_fingerprint, validate_task_fingerprint, write_task_fingerprint,
)


class TaskFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="poet-task-fingerprint-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.asset = self.root / "robot.usda"
        self.asset.write_text("test robot asset")
        self.sidecar = self.root / "task_fingerprint.json"

    def build(self, **changes):
        arguments = dict(episode_steps=2000, solver="featherstone", substeps=16,
                         simulation_versions={"newton": "1.6.1", "source_commit": "abc"},
                         task_config={"physics_dt": 1 / 120, "decimation": 2},
                         source_files={"robot_usd": self.asset})
        arguments.update(changes)
        return build_task_fingerprint(**arguments)

    def test_round_trip_and_mapping_order_independence(self):
        fingerprint = self.build()
        write_task_fingerprint(self.sidecar, fingerprint)
        equivalent = self.build(simulation_versions={"source_commit": "abc", "newton": "1.6.1"},
                                task_config={"decimation": 2, "physics_dt": 1 / 120})
        self.assertEqual(fingerprint, equivalent)
        self.assertIsNone(validate_task_fingerprint(self.sidecar, equivalent))
        self.assertEqual(list(self.root.glob(".task_fingerprint_*.tmp")), [])

    def test_rejects_different_task_horizon_solver_dynamics_or_versions(self):
        write_task_fingerprint(self.sidecar, self.build())
        cases = [
            (dict(episode_steps=1000), "episode_steps"),
            (dict(solver="mjwarp"), "solver"),
            (dict(substeps=1), "substeps"),
            (dict(task_config={"physics_dt": 1 / 60, "decimation": 2}), "task_config.physics_dt"),
            (dict(simulation_versions={"newton": "1.7.0", "source_commit": "abc"}), "simulation_versions.newton"),
        ]
        for changes, expected_name in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, expected_name):
                validate_task_fingerprint(self.sidecar, self.build(**changes))

    def test_asset_content_matters_and_absolute_location_does_not(self):
        fingerprint = self.build()
        copied_asset = self.root / "elsewhere.usda"
        copied_asset.write_bytes(self.asset.read_bytes())
        self.assertEqual(fingerprint, self.build(source_files={"robot_usd": copied_asset}))
        write_task_fingerprint(self.sidecar, fingerprint)
        self.asset.write_text("different robot mass")
        with self.assertRaisesRegex(ValueError, "source_sha256.robot_usd"):
            validate_task_fingerprint(self.sidecar, self.build())

    def test_missing_corrupt_or_tampered_sidecar_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_task_fingerprint(self.sidecar, self.build())
        self.sidecar.write_text("not json")
        with self.assertRaisesRegex(ValueError, "Cannot read"):
            validate_task_fingerprint(self.sidecar, self.build())
        tampered = self.build()
        tampered["payload"]["episode_steps"] = 100
        self.sidecar.write_text(json.dumps(tampered))
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_task_fingerprint(self.sidecar, self.build())

    def test_bad_configuration_is_rejected(self):
        for changes in (dict(episode_steps=0), dict(substeps=True), dict(solver=""),
                        dict(task_config={"physics_dt": float("nan")})):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.build(**changes)


if __name__ == "__main__":
    unittest.main()
