import os
import sqlite3
import tempfile
import unittest

import numpy as np

from services.pipelines.post_processing_pipeline import process_im_arrays
from services.storage.sqlite_shards import (
    SqliteArrayShardWriter,
    list_signatures,
    prune_final_states_below_min_events,
)


class GlobalFinalStateThresholdTests(unittest.TestCase):
    def test_post_processing_applies_one_threshold_to_all_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(input_dir)
            paths = [
                os.path.join(input_dir, f"im_batch_{index}.sqlite")
                for index in (1, 2)
            ]
            for index, path in enumerate(paths, start=1):
                writer = SqliteArrayShardWriter(path)
                writer.record_final_state_count("2e_0m_0j_0g_0t_0b", 6)
                writer.append_array(
                    f"parsed_batch{index}_FS_2e_0m_0j_0g_0t_0b_IM_e0e1",
                    np.array([100.0 + index]),
                )
                writer.record_final_state_count("0e_2m_0j_0g_0t_0b", 4)
                writer.append_array(
                    f"parsed_batch{index}_FS_0e_2m_0j_0g_0t_0b_IM_m0m1",
                    np.array([200.0 + index]),
                )
                writer.close()

            outputs = process_im_arrays(
                {
                    "input_dir": input_dir,
                    "output_dir": output_dir,
                    "peak_detection_bin_width_gev": 10.0,
                    "z_peak_cutoff": 0.0,
                    "max_mass_cutoff": 10_000.0,
                    "min_events_per_fs": 10,
                    "batch_job_index": None,
                }
            )

            self.assertEqual(outputs, ["processed_batch_1.sqlite"])
            self.assertTrue(
                all(
                    all("_FS_0e_2m_" not in sig for sig in list_signatures(path))
                    for path in paths
                )
            )
            output_signatures = list_signatures(os.path.join(output_dir, outputs[0]))
            self.assertEqual(len(output_signatures), 1)
            self.assertIn("_FS_2e_0m_", output_signatures[0])

    def test_counts_are_aggregated_across_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "im.sqlite")
            writer = SqliteArrayShardWriter(path)
            writer.append_array(
                "parsed_chunk1_FS_2e_0m_0j_0g_0t_0b_IM_e0e1", np.arange(6)
            )
            writer.append_array(
                "parsed_chunk2_FS_2e_0m_0j_0g_0t_0b_IM_e0e1", np.arange(6)
            )
            writer.append_array(
                "parsed_chunk1_FS_0e_2m_0j_0g_0t_0b_IM_m0m1", np.arange(4)
            )
            writer.close()

            removed = prune_final_states_below_min_events(path, 10)

            self.assertEqual(removed, ["_FS_0e_2m_0j_0g_0t_0b"])
            signatures = list_signatures(path)
            self.assertEqual(len(signatures), 2)
            self.assertTrue(all("_FS_2e_" in sig for sig in signatures))

    def test_pre_cut_counts_are_aggregated_across_batch_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [os.path.join(tmpdir, f"im_batch_{i}.sqlite") for i in (1, 2)]
            for index, path in enumerate(paths, start=1):
                writer = SqliteArrayShardWriter(path)
                writer.record_final_state_count("2e_0m_0j_0g_0t_0b", 6)
                writer.append_array(
                    f"parsed_batch{index}_FS_2e_0m_0j_0g_0t_0b_IM_e0e1",
                    np.arange(1),
                )
                writer.close()

            removed = prune_final_states_below_min_events(paths, 10)

            self.assertEqual(removed, [])
            self.assertTrue(all(len(list_signatures(path)) == 1 for path in paths))

    def test_underpopulated_state_is_removed_from_every_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [os.path.join(tmpdir, f"im_batch_{i}.sqlite") for i in (1, 2)]
            for index, path in enumerate(paths, start=1):
                writer = SqliteArrayShardWriter(path)
                writer.record_final_state_count("2e_0m_0j_0g_0t_0b", 4)
                writer.append_array(
                    f"parsed_batch{index}_FS_2e_0m_0j_0g_0t_0b_IM_e0e1",
                    np.arange(4),
                )
                writer.close()

            removed = prune_final_states_below_min_events(paths, 10)

            self.assertEqual(removed, ["_FS_2e_0m_0j_0g_0t_0b"])
            self.assertTrue(all(list_signatures(path) == [] for path in paths))

    def test_mixed_legacy_and_counted_shards_are_both_counted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_path = os.path.join(tmpdir, "im_batch_1.sqlite")
            writer = SqliteArrayShardWriter(legacy_path)
            writer.append_array(
                "parsed_batch1_FS_2e_0m_0j_0g_0t_0b_IM_e0e1", np.arange(6)
            )
            writer.close()
            # Simulate a shard written before final_state_counts was added.
            conn = sqlite3.connect(legacy_path)
            try:
                conn.execute("DROP TABLE final_state_counts")
                conn.commit()
            finally:
                conn.close()

            counted_path = os.path.join(tmpdir, "im_batch_2.sqlite")
            writer = SqliteArrayShardWriter(counted_path)
            writer.record_final_state_count("2e_0m_0j_0g_0t_0b", 6)
            writer.append_array(
                "parsed_batch2_FS_2e_0m_0j_0g_0t_0b_IM_e0e1", np.arange(1)
            )
            writer.close()

            removed = prune_final_states_below_min_events(
                [legacy_path, counted_path], 10
            )

            self.assertEqual(removed, [])


if __name__ == "__main__":
    unittest.main()
