import unittest
from unittest import mock

from services.parsing.root_io import open_root_file


class XRootDSourceSelectionTests(unittest.TestCase):
    @mock.patch("services.parsing.root_io.uproot.open")
    def test_root_url_uses_native_xrootd_source(self, mocked_open):
        url = "root://eospublic.cern.ch:1094//eos/example.root"

        open_root_file(url, timeout=60)

        args, kwargs = mocked_open.call_args
        self.assertEqual(args, (url,))
        self.assertEqual(kwargs["timeout"], 60)
        self.assertEqual(kwargs["handler"].__name__, "XRootDSource")

    @mock.patch("services.parsing.root_io.uproot.open")
    def test_local_path_keeps_uproot_default_source(self, mocked_open):
        path = "parsed_data/example.root"

        open_root_file(path)

        mocked_open.assert_called_once_with(path)

    @mock.patch("services.parsing.root_io.uproot.open")
    def test_explicit_handler_is_not_overridden(self, mocked_open):
        sentinel_handler = object()
        url = "root://eospublic.cern.ch:1094//eos/example.root"

        open_root_file(url, handler=sentinel_handler)

        mocked_open.assert_called_once_with(url, handler=sentinel_handler)


if __name__ == "__main__":
    unittest.main()
