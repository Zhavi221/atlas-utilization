"""Tests for trigger branch extraction configuration."""

import unittest

from services.parsing.file_parser import FileParser
from services.parsing import schemas


def _tree_branches_with_trigger_metadata() -> set[str]:
    return {
        "AnalysisElectronsAuxDyn.pt",
        "AnalysisElectronsAuxDyn.eta",
        "AnalysisElectronsAuxDyn.phi",
        schemas.get_all_trigger_branches()[0],
        schemas.RANDOM_RUN_NUMBER_BRANCH,
    }


class TriggerConfigWiringTests(unittest.TestCase):
    def test_trigger_metadata_is_not_requested_when_matching_is_disabled(self):
        branches = FileParser._extract_branches_by_schema(
            _tree_branches_with_trigger_metadata(),
            "2024r-pp",
            enable_trigger_matching=False,
        )

        self.assertNotIn("_triggerMatch", branches)
        self.assertNotIn("_runNumber", branches)

    def test_trigger_metadata_is_requested_when_matching_is_enabled(self):
        tree_branches = _tree_branches_with_trigger_metadata()
        trigger_branch = schemas.get_all_trigger_branches()[0]

        branches = FileParser._extract_branches_by_schema(
            tree_branches,
            "2024r-pp",
            enable_trigger_matching=True,
        )

        self.assertEqual(branches["_triggerMatch"], {trigger_branch: trigger_branch})
        self.assertEqual(
            branches["_runNumber"],
            {schemas.RANDOM_RUN_NUMBER_BRANCH: "_runNumber"},
        )
