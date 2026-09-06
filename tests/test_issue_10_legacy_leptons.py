import unittest

import awkward as ak

from services.parsing.file_parser import FileParser
from services.parsing.schemas import get_schema_for_release


class LegacyCombinedLeptonTests(unittest.TestCase):
    def _combined_leptons(self):
        return ak.Array([
            [
                {"pt": 50.0, "eta": 0.1, "phi": 0.2, "type": 11, "charge": -1},
                {"pt": 40.0, "eta": -0.1, "phi": -0.2, "type": -13, "charge": 1},
                {"pt": 30.0, "eta": 0.3, "phi": 0.4, "type": -11, "charge": 1},
            ]
        ])

    def test_combined_collection_is_split_by_absolute_pdg_id(self):
        combined = self._combined_leptons()
        objects = {"Electrons": combined, "Muons": combined}

        result = FileParser._split_combined_leptons(objects, "2016e-8tev")

        self.assertEqual(ak.to_list(result["Electrons"]["type"]), [[11, -11]])
        self.assertEqual(ak.to_list(result["Muons"]["type"]), [[-13]])

    def test_mc_suffix_uses_same_legacy_split(self):
        combined = self._combined_leptons()
        objects = {"Electrons": combined, "Muons": combined}

        result = FileParser._split_combined_leptons(
            objects, "2025e-13tev-beta_mc"
        )

        self.assertEqual(ak.to_list(result["Electrons"]["type"]), [[11, -11]])
        self.assertEqual(ak.to_list(result["Muons"]["type"]), [[-13]])

    def test_legacy_schema_reads_identity_fields(self):
        schema = get_schema_for_release("2016e-8tev")

        self.assertIn("type", schema["objects"]["Electrons"])
        self.assertIn("charge", schema["objects"]["Electrons"])
        self.assertIn("type", schema["objects"]["Muons"])

    def test_nonlegacy_collection_is_unchanged(self):
        electrons = self._combined_leptons()
        objects = {"Electrons": electrons}

        result = FileParser._split_combined_leptons(objects, "2024r-pp")

        self.assertIs(result, objects)
        self.assertIs(result["Electrons"], electrons)


if __name__ == "__main__":
    unittest.main()
