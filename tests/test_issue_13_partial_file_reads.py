import unittest
from unittest.mock import patch

import awkward as ak

from services.parsing.file_parser import FileParser, PartialFileReadError
from services.parsing.threaded_processor import (
    ParsingStatisticsCollector,
    ThreadedFileProcessor,
)


BRANCH_MAPPING = {
    "Electrons": {
        "E.pt": "pt",
        "E.eta": "eta",
        "E.phi": "phi",
    }
}


class RootBatchPrefixTests(unittest.TestCase):
    def test_stops_at_failure_and_returns_completed_prefix(self):
        class FakeTree:
            def __init__(self):
                self.starts = []

            def arrays(self, branches, entry_start, entry_stop, library):
                self.starts.append(entry_start)
                if entry_start == 2:
                    raise OSError("corrupt basket")
                return ak.Array({
                    "E.pt": [[1.0], [2.0]],
                    "E.eta": [[0.0], [0.0]],
                    "E.phi": [[0.0], [0.0]],
                })

        tree = FakeTree()
        objects, error = FileParser._read_file_in_batches(
            tree,
            set(BRANCH_MAPPING["Electrons"]),
            BRANCH_MAPPING,
            n_entries=6,
            batch_size=2,
        )

        self.assertEqual(tree.starts, [0, 2])
        self.assertEqual(len(objects["Electrons"]), 2)
        self.assertIn("batch 2-4", str(error))
        self.assertIn("OSError: corrupt basket", str(error))

    def test_opened_file_exposes_partial_events_with_failure_reason(self):
        class FakeTree:
            num_entries = 4

            def keys(self):
                return set(BRANCH_MAPPING["Electrons"])

            def arrays(self, branches, entry_start, entry_stop, library):
                if entry_start == 2:
                    raise OSError("corrupt basket")
                return ak.Array({
                    "E.pt": [[1.0], [2.0]],
                    "E.eta": [[0.0], [0.0]],
                    "E.phi": [[0.0], [0.0]],
                })

        class FakeRoot(dict):
            def keys(self):
                return ["events;1"]

        root = FakeRoot(events=FakeTree())
        with (
            patch.object(
                FileParser,
                "_extract_branches_by_schema",
                return_value=BRANCH_MAPPING,
            ),
            patch.object(
                FileParser,
                "_filter_accessible_branches",
                return_value=BRANCH_MAPPING,
            ),
            self.assertRaises(PartialFileReadError) as raised,
        ):
            FileParser._parse_opened_file(
                root,
                ["events"],
                "2024r-pp",
                2,
                "broken.root",
                False,
                None,
            )

        self.assertEqual(len(raised.exception.events), 2)
        self.assertIn("batch 2-4", str(raised.exception))
        self.assertIn("corrupt basket", str(raised.exception))


class PartialFailureAccountingTests(unittest.TestCase):
    def test_partial_events_are_yielded_but_file_is_recorded_failed(self):
        events = ak.zip({
            "Electrons": ak.Array([[{"pt": 1.0}], [{"pt": 2.0}]])
        }, depth_limit=1)

        class PartialParser:
            def parse_file(self, file_path, **kwargs):
                cause = RuntimeError("batch 2-4 failed with OSError: corrupt basket")
                raise PartialFileReadError(file_path, events, cause)

        successes = []
        failures = []
        processor = ThreadedFileProcessor(PartialParser(), 1, show_progress=False)

        batches = list(processor.process_files(
            ["broken.root"],
            ["events"],
            "2024r-pp",
            on_success=lambda *args: successes.append(args),
            on_error=lambda url, error: failures.append((url, error)),
        ))

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].event_count, 2)
        self.assertEqual(successes, [])
        self.assertEqual(len(failures), 1)
        self.assertIn("retaining 2 events", str(failures[0][1]))

        stats = ParsingStatisticsCollector()
        stats.record_failure(*failures[0])
        summary = stats.get_summary()
        self.assertEqual(summary["successful_files"], 0)
        self.assertEqual(summary["failed_files"], 1)
        self.assertEqual(summary["total_events"], 2)

    def test_total_failure_is_recorded_without_a_batch(self):
        class NullParser:
            def parse_file(self, *args, **kwargs):
                return None

        failures = []
        processor = ThreadedFileProcessor(NullParser(), 1, show_progress=False)

        batches = list(processor.process_files(
            ["unreadable.root"],
            ["events"],
            "2024r-pp",
            on_error=lambda url, error: failures.append((url, error)),
        ))

        self.assertEqual(batches, [])
        self.assertEqual(len(failures), 1)
        self.assertIn("no event data", str(failures[0][1]))


if __name__ == "__main__":
    unittest.main()
