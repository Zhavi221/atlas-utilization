import unittest

from orchestration.handlers.parsing_handler import select_metadata_for_parsing
from services.metadata.fetcher import MetadataFetcher, UrlType, _classify_url


DATA_2024 = "https://example.test/rucio/data23_13p6TeV/file.root"
MC_2024 = "https://example.test/rucio/mc23_13p6TeV/file.root"
DATA_2020 = "https://example.test/rucio/data15_13TeV/file.root"
MC_2020 = "https://example.test/rucio/mc20_13TeV/file.root"


class MetadataModeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.metadata = MetadataFetcher._separate_mc_files(
            {
                "2024r-pp": [DATA_2024, MC_2024],
                "2020e-13tev": [DATA_2020, MC_2020],
            }
        )

    def test_parse_mc_true_selects_only_mc_for_requested_release(self):
        selected = select_metadata_for_parsing(
            self.metadata, ["2024r-pp"], parse_mc=True
        )

        self.assertEqual(selected, {"2024r-pp_mc": [MC_2024]})
        self.assertTrue(
            all(
                _classify_url(url) is UrlType.MC
                for urls in selected.values()
                for url in urls
            )
        )

    def test_parse_mc_false_selects_only_data_for_requested_release(self):
        selected = select_metadata_for_parsing(
            self.metadata, ["2024r-pp"], parse_mc=False
        )

        self.assertEqual(selected, {"2024r-pp": [DATA_2024]})
        self.assertTrue(
            all(
                _classify_url(url) is UrlType.DATA
                for urls in selected.values()
                for url in urls
            )
        )

    def test_mc_suffix_in_config_is_normalized_to_selected_mode(self):
        mc_selected = select_metadata_for_parsing(
            self.metadata, ["2024r-pp_mc"], parse_mc=True
        )
        data_selected = select_metadata_for_parsing(
            self.metadata, ["2024r-pp_mc"], parse_mc=False
        )

        self.assertEqual(mc_selected, {"2024r-pp_mc": [MC_2024]})
        self.assertEqual(data_selected, {"2024r-pp": [DATA_2024]})

    def test_unrequested_releases_are_excluded(self):
        selected = select_metadata_for_parsing(
            self.metadata, ["2024r-pp"], parse_mc=True
        )

        self.assertNotIn("2020e-13tev_mc", selected)
        self.assertNotIn("2020e-13tev", selected)

    def test_missing_requested_mode_fails_instead_of_using_other_mode(self):
        with self.assertRaisesRegex(RuntimeError, "No MC metadata found"):
            select_metadata_for_parsing(
                {"2024r-pp": [DATA_2024]},
                ["2024r-pp"],
                parse_mc=True,
            )


if __name__ == "__main__":
    unittest.main()
