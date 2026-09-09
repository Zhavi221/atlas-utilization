import unittest

import awkward as ak

from services.calculations.physics_calcs import (
    filter_events_by_kinematics,
    filter_events_by_particle_counts,
)


def _particles(counts):
    values = [
        {"pt": 50.0, "eta": 0.0, "phi": 0.0}
        for count in counts
        for _ in range(count)
    ]
    return ak.unflatten(ak.Array(values), counts)


class MissingCollectionSelectionTests(unittest.TestCase):
    def test_missing_required_collection_rejects_event(self):
        events = ak.zip({"BJets": _particles([1])}, depth_limit=1)

        selected = filter_events_by_particle_counts(
            events,
            {"Electrons": {"min": 1, "max": 4}},
            is_particle_counts_range=True,
        )

        self.assertEqual(len(selected), 0)

    def test_missing_optional_collection_does_not_reject_event(self):
        events = ak.zip({"BJets": _particles([1])}, depth_limit=1)

        selected = filter_events_by_particle_counts(
            events,
            {"Electrons": {"min": 0, "max": 4}},
            is_particle_counts_range=True,
        )

        self.assertEqual(len(selected), 1)

    def test_requested_isolation_must_exist(self):
        events = ak.zip({"Electrons": _particles([1])}, depth_limit=1)

        with self.assertRaisesRegex(ValueError, "requires missing field"):
            filter_events_by_kinematics(
                events,
                {"Electrons": {"rel_isolation_max": 0.06}},
            )

    def test_requested_kinematic_field_must_exist(self):
        particles = ak.Array([[{"pt": 50.0, "phi": 0.0}]])
        events = ak.zip({"Electrons": particles}, depth_limit=1)

        with self.assertRaisesRegex(ValueError, "missing configured kinematic field 'eta'"):
            filter_events_by_kinematics(
                events,
                {"Electrons": {"eta": {"min": -2.5, "max": 2.5}}},
            )


if __name__ == "__main__":
    unittest.main()
