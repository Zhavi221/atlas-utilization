"""Regression tests for the mass-calculation object allow-list."""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import awkward as ak
import numpy as np
import uproot
import yaml

from domain.config import PipelineConfig
from orchestration.handlers.mass_calculation_handler import MassCalculationHandler
from pipeline.executor import PipelineExecutor
from services.parsing.file_parser import FileParser


class ObjectAllowListTests(unittest.TestCase):
    def test_particle_counts_reject_excluded_object_even_with_zero_maximum(self):
        config = {
            "tasks": {"do_parsing": True},
            "parsing_task_config": {
                "output_path": "out", "file_urls_path": "urls", "jobs_logs_path": "logs",
                "particle_counts": {"photons": {"min": 0, "max": 0}},
            },
            "mass_calculation_task_config": {
                "objects_to_calculate": ["Electrons", "Muons"],
            },
        }

        with self.assertRaisesRegex(ValueError, "objects_to_calculate.*photons"):
            PipelineConfig.from_dict(config)

    def test_parser_retains_only_requested_physics_objects(self):
        events = {
            "Electrons": ak.Array([[{"pt": 42.0}]]),
            "Photons": ak.Array([[{"pt": 17.0}]]),
            "_runNumber": ak.Array([300000]),
        }

        filtered = FileParser._retain_requested_objects(events, ("Electrons",))

        self.assertEqual(set(filtered), {"Electrons", "_runNumber"})

    def test_branch_selection_does_not_request_excluded_objects(self):
        schema = {
            "objects": {
                "Electrons": ["pt", "eta", "phi"],
                "Photons": ["pt", "eta", "phi"],
            },
            "branch_prefix": "",
            "branch_suffix": "",
            "naming_pattern": "flat",
        }
        branches = {
            "Electrons_pt", "Electrons_eta", "Electrons_phi",
            "Photons_pt", "Photons_eta", "Photons_phi",
        }

        with patch("services.parsing.file_parser.schemas.get_schema_for_release", return_value=schema):
            selected = FileParser._extract_branches_by_schema(
                branches, "test-release", objects_to_parse=("Electrons",)
            )

        self.assertEqual(set(selected), {"Electrons"})
        self.assertNotIn("Photons_pt", selected["Electrons"])

    def test_mass_loader_reconstruction_excludes_unrequested_fields(self):
        # The reconstruction filtering is what protects histogram/final-state
        # counts when mass calculation is run on an older parsed ROOT file.
        class FakeTree:
            def keys(self):
                return ["nElectrons", "nPhotons", "Electrons_pt", "Photons_pt"]

            def __getitem__(self, name):
                return type("Branch", (), {"array": lambda _self, library: ak.Array([[1.0]])})()

        arrays = MassCalculationHandler._reconstruct_particle_arrays(
            FakeTree(), ("Electrons",)
        )

        self.assertEqual(arrays.fields, ["Electrons"])

    def test_default_config_has_no_count_filter_for_excluded_objects(self):
        config_path = Path(__file__).parents[1] / "config.yaml"
        config = PipelineConfig.from_dict(yaml.safe_load(config_path.read_text()))

        self.assertEqual(config.parsing_config.objects_to_parse, (
            "Electrons", "Muons", "Jets", "BJets"
        ))
        self.assertEqual(set(config.parsing_config.particle_counts), {
            "electrons", "muons", "jets", "bjets"
        })

    def test_object_statistics_ignore_excluded_branches_in_existing_parsed_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parsed_file = Path(temp_dir) / "parsed.root"
            with uproot.recreate(parsed_file) as root_file:
                root_file["events"] = {
                    "nElectrons": np.array([1, 2]),
                    "nPhotons": np.array([8, 13]),
                }

            # This helper only uses its logger if a file cannot be read.
            executor = object.__new__(PipelineExecutor)
            _, particle_stats = executor._read_parsed_data_stats(
                temp_dir, {"Electrons"}
            )

        self.assertEqual(particle_stats["particle_counts"], {"Electrons": 3})


if __name__ == "__main__":
    unittest.main()
