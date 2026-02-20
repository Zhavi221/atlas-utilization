"""
Invariant Mass Calculator.

Provides a clean interface for calculating invariant masses from particle events.
Supports grouping by final state, filtering, and batch processing.
"""
import awkward as ak
import vector
from typing import Dict, Iterator, List
from collections import Counter

from services.calculations import consts, physics_calcs


class IMCalculator:
    def __init__(self, events: ak.Array, min_events_per_fs: int,
                 min_k: int, max_k: int, min_n: int, max_n: int):
        self.events = events
        self.min_events_per_fs = min_events_per_fs
        self.min_k = min_k
        self.max_k = max_k
        self.min_n = min_n
        self.max_n = max_n
        self._all_events_fs = None
        vector.register_awkward()

    def calculate_invariant_mass(self, particle_events: ak.Array) -> ak.Array:
        if len(particle_events) == 0:
            return ak.Array([])

        all_vectors = self._concat_particles_to_vectors(particle_events)
        combined_vectors = ak.concatenate(all_vectors, axis=1)
        total_momentum = ak.sum(combined_vectors, axis=1)

        if hasattr(total_momentum, 'tau'):
            return total_momentum.tau
        return total_momentum.mass

    def _concat_particles_to_vectors(self, particle_events: ak.Array) -> List[ak.Array]:
        all_vectors = []
        for particle_type in particle_events.fields:
            particle_array = particle_events[particle_type]
            mass = self._get_particle_mass(particle_type, particle_array)
            momentum_vector = vector.zip({
                "pt": particle_array.pt,
                "phi": particle_array.phi,
                "eta": particle_array.eta,
                "mass": mass
            })
            all_vectors.append(momentum_vector)
        return all_vectors

    @staticmethod
    def _get_particle_mass(particle_type: str, particle_array: ak.Array) -> ak.Array:
        if 'm' in particle_array.fields:
            return particle_array.m
        return consts.KNOWN_MASSES.get(particle_type, 0.0)

    def _get_all_events_fs(self) -> ak.Array:
        if self._all_events_fs is None:
            particle_counts = ak.num(self.events)
            num_events = len(self.events)
            zero_array = ak.Array([0] * num_events) if num_events > 0 else ak.Array([])

            e = ak.to_numpy(getattr(particle_counts, "Electrons", zero_array))
            m = ak.to_numpy(getattr(particle_counts, "Muons", zero_array))
            j = ak.to_numpy(getattr(particle_counts, "Jets", zero_array))
            g = ak.to_numpy(getattr(particle_counts, "Photons", zero_array))

            all_events_fs = [
                f"{e}e_{m}m_{j}j_{g}g"
                for e, m, j, g in zip(e, m, j, g)
                if self._is_valid_fs([e, m, j, g])
            ]
            self._all_events_fs = ak.Array(all_events_fs)
        return self._all_events_fs

    def _is_valid_fs(self, particle_counts) -> bool:
        total_types = [p for p in particle_counts if p > 0]
        if len(total_types) < self.min_n or len(total_types) > self.max_n:
            return False
        for p in total_types:
            if p < self.min_k or p > self.max_k:
                return False
        return True

    def group_by_final_state(self) -> Iterator[str]:
        all_events_fs = self._get_all_events_fs()
        all_events_fs_list = ak.to_list(all_events_fs)

        fs_by_count = Counter(all_events_fs_list)
        fs_by_count_sorted = [
            (fs, count) for fs, count in fs_by_count.most_common()
            if count >= self.min_events_per_fs
        ]

        for fs, _count in fs_by_count_sorted:
            yield self._limit_particles_in_fs(fs, threshold=4)

    def get_events_for_final_state(self, final_state: str) -> ak.Array:
        all_events_fs = self._get_all_events_fs()
        mask = (all_events_fs == final_state)
        return self.events[mask]

    @staticmethod
    def _limit_particles_in_fs(final_state: str, threshold: int = 4) -> str:
        fs_particles = final_state.split('_')
        for str_amount_particle in fs_particles:
            if len(str_amount_particle) < 2:
                continue
            amount_to_calc = str_amount_particle[0]
            particle_letter = str_amount_particle[1]
            if amount_to_calc.isdigit():
                amount = int(amount_to_calc)
                if amount > threshold:
                    final_state = final_state.replace(
                        f"{amount}{particle_letter}", f"{threshold}{particle_letter}")
        return final_state

    @staticmethod
    def does_final_state_contain_combination(final_state: str, combination: Dict[str, int]) -> bool:
        fs_particles = final_state.split('_')
        for str_amount_particle in fs_particles:
            if len(str_amount_particle) < 2:
                continue
            amount_to_calc = str_amount_particle[0]
            particle_letter = str_amount_particle[1]

            if not amount_to_calc.isdigit():
                continue

            particle = consts.LETTER_PARTICLE_MAPPING.get(particle_letter)
            if particle is None or particle not in combination:
                continue

            fs_particle_amount = int(amount_to_calc)
            required_amount = combination[particle]
            if fs_particle_amount < required_amount:
                return False
        return True

    def filter_by_particle_counts(self, events, particle_counts,
                                  is_exact_count=False, is_particle_counts_range=False):
        return physics_calcs.filter_events_by_particle_counts(
            events, particle_counts, is_exact_count, is_particle_counts_range)

    def filter_by_kinematics(self, events, kinematic_cuts):
        return physics_calcs.filter_events_by_kinematics(events, kinematic_cuts)

    def slice_by_field(self, events, particle_counts, field_to_slice_by="pt"):
        return physics_calcs.slice_events_by_field(
            events, particle_counts, field_to_slice_by)
