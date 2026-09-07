import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pipeline.executor import PipelineExecutor
from services.storage.sqlite_shards import SqliteArrayShardWriter


class MassCalculationStatsTests(unittest.TestCase):
    def setUp(self):
        self.executor = PipelineExecutor.__new__(PipelineExecutor)
        self.executor.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _write_shard(path: Path, elapsed: float) -> None:
        writer = SqliteArrayShardWriter(str(path))
        writer.append_array(
            "sample_FS_2e_0m_0j_0g_0t_0b_IM_e0e1",
            np.array([10.0, 20.0]),
        )
        writer.set_metadata("mass_calculation_time_sec", elapsed)
        writer.close()

    def test_reads_and_sums_timing_from_sqlite_shard_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            im_dir = Path(tmp_dir)
            self._write_shard(im_dir / "im_batch_1.sqlite", 1.25)
            self._write_shard(im_dir / "im_batch_2.sqlite", 2.75)

            stats = self.executor._read_im_array_stats(str(im_dir))

            self.assertEqual(stats["total_time_sec"], 4.0)
            self.assertEqual(stats["total_mass_values"], 4)
            self.assertEqual(stats["total_combinations"], 1)

    def test_legacy_shard_without_timing_metadata_remains_readable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            im_dir = Path(tmp_dir)
            shard_path = im_dir / "im_batch_1.sqlite"
            self._write_shard(shard_path, 3.0)

            import sqlite3

            with sqlite3.connect(str(shard_path)) as conn:
                conn.execute("DROP TABLE shard_metadata")

            stats = self.executor._read_im_array_stats(str(im_dir))

            self.assertEqual(stats["total_time_sec"], 0.0)
            self.assertEqual(stats["total_mass_values"], 2)

    def test_reads_structured_stage_timings_without_logs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            context = SimpleNamespace(
                parsing_stats=None,
                custom_data={"mass_calc": {"total_time_sec": 6.5}},
                config=SimpleNamespace(batch_job_index=2),
            )
            self.executor.save_stage_stats(str(run_dir), context)

            # A conflicting human-readable log must not affect the result.
            (run_dir / "logs" / "batch_2.out").write_text(
                "Mass calculation complete: changed wording in 999.0s\n"
            )
            timings = self.executor._read_stage_timings(str(run_dir))

            self.assertEqual(timings["mass_calc"], 6.5)


if __name__ == "__main__":
    unittest.main()
