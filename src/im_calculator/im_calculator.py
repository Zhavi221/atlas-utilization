"""
Invariant Mass Calculator

Provides a clean interface for calculating invariant masses from particle events.
Supports grouping by final state, filtering, and batch processing.
"""
import awkward as ak
import vector
from typing import Dict, Iterator, Tuple, Optional, List
from collections import Counter

from src.calculations import consts, physics_calcs


class IMCalculator:
    """
    Calculator for invariant mass computations from particle events.
    
    This class provides methods to:
    - Calculate invariant masses for particle combinations
    - Group events by final state
    - Filter events by particle counts and kinematics
    - Process events in batches for memory efficiency
    """
    
    def __init__(self, events: ak.Array):
        """
        Initialize the calculator with particle events.
        
        Args:
            events: Awkward array containing particle events with fields like
                   Electrons, Muons, Jets, Photons
        """
        self.events = events
        vector.register_awkward()
    
    def calculate_invariant_mass(self, particle_events: ak.Array) -> ak.Array:
        """
        Calculate invariant mass for a set of particle events.
        
        Args:
            particle_events: Awkward array with particle fields (e.g., Electrons, Muons)
            
        Returns:
            Array of invariant masses (one per event)
        """
        if len(particle_events) == 0:
            return ak.Array([])

        all_vectors = self._concat_particles_to_vectors(particle_events)
        combined_vectors = ak.concatenate(all_vectors, axis=1)

        total_momentum = ak.sum(combined_vectors, axis=1)
        
        if hasattr(total_momentum, 'tau'):
            inv_mass = total_momentum.tau
        else:
            inv_mass = total_momentum.mass

        return inv_mass
    
    def _concat_particles_to_vectors(self, particle_events: ak.Array) -> List[ak.Array]:
        """
        Convert particle events to momentum vectors.
        
        Args:
            particle_events: Awkward array with particle fields
            
        Returns:
            List of momentum vector arrays, one per particle type
        """
        all_vectors = []
        particle_types = particle_events.fields
        
        for particle_type in particle_types:
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
        """
        Get particle mass from array or use known mass constant.
        
        Args:
            particle_type: Name of particle type (e.g., "Electrons")
            particle_array: Array of particles
            
        Returns:
            Array of masses
        """
        if 'm' in particle_array.fields:
            return particle_array.m
        else:
            return consts.KNOWN_MASSES.get(particle_type, 0.0)
    
    def group_by_final_state(self) -> Iterator[Tuple[str, ak.Array]]:
        """
        Group events by their final state (particle counts).
        
        Yields:
            Tuples of (final_state_string, events_matching_final_state)
            Final state format: "{e}e_{m}m_{j}j_{p}p" where numbers are counts
            
        Note:
            If a particle type doesn't exist in the events, its count defaults to 0.
        """
        particle_counts = ak.num(self.events)
        
        # Safely get particle counts, defaulting to 0 if particle type doesn't exist
        # This handles cases where certain particle types aren't present in the data
        # Create zero array with same length as events for missing particle types
        num_events = len(self.events)
        zero_array = ak.Array([0] * num_events) if num_events > 0 else ak.Array([])
        
        e = getattr(particle_counts, "Electrons", zero_array)
        m = getattr(particle_counts, "Muons", zero_array)
        j = getattr(particle_counts, "Jets", zero_array)
        p = getattr(particle_counts, "Photons", zero_array)
        
        all_events_fs = [f"{e}e_{m}m_{j}j_{p}p" for e, m, j, p in zip(e, m, j, p)]
        fs_by_count = Counter(all_events_fs)
        fs_by_count_sorted = fs_by_count.most_common()
        
        for fs, count in fs_by_count_sorted:
            mask = (ak.Array(all_events_fs) == fs)
            events_matching_fs = self.events[mask]
            fs_limited = self._limit_particles_in_fs(fs, threshold=4)
            yield (fs_limited, events_matching_fs)
    
    @staticmethod
    def _limit_particles_in_fs(final_state: str, threshold: int = 4) -> str:
        """
        Limit particle counts in final state string to threshold.
        
        Args:
            final_state: Final state string like "2e_3m_5j_1p"
            threshold: Maximum count to allow
            
        Returns:
            Modified final state string with counts capped at threshold
        """
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
        """
        Check if a final state contains enough particles for a combination.
        
        Args:
            final_state: Final state string like "2e_3m_5j_1p"
            combination: Dictionary mapping particle types to required counts
                        e.g., {"Electrons": 2, "Jets": 3}
        
        Returns:
            True if final state has enough particles for the combination
        """
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
    
    def filter_by_particle_counts(
        self, 
        events: ak.Array, 
        particle_counts: Dict[str, int],
        is_exact_count: bool = False,
        is_particle_counts_range: bool = False
    ) -> ak.Array:
        """
        Filter events by particle counts.
        
        Args:
            events: Events to filter
            particle_counts: Dictionary of particle type to count/range
            is_exact_count: If True, require exact count match
            is_particle_counts_range: If True, particle_counts contains min/max dicts
            
        Returns:
            Filtered events array
        """
        return physics_calcs.filter_events_by_particle_counts(
            events, particle_counts, is_exact_count, is_particle_counts_range
        )
    
    def filter_by_kinematics(
        self, 
        events: ak.Array, 
        kinematic_cuts: Dict[str, Dict[str, float]]
    ) -> ak.Array:
        """
        Filter events by kinematic cuts (pt, eta, phi).
        
        Args:
            events: Events to filter
            kinematic_cuts: Dictionary with keys like "pt", "eta", "phi"
                          containing min/max values
                          
        Returns:
            Filtered events array
        """
        return physics_calcs.filter_events_by_kinematics(events, kinematic_cuts)
    
    def slice_by_field(
        self, 
        events: ak.Array, 
        particle_counts: Dict[str, int],
        field_to_slice_by: str = "pt"
    ) -> ak.Array:
        """
        Slice events to keep only top N particles by a field (e.g., pt).
        
        Args:
            events: Events to slice
            particle_counts: Dictionary mapping particle types to counts to keep
            field_to_slice_by: Field name to sort by (default: "pt")
            
        Returns:
            Sliced events array
        """
        return physics_calcs.slice_events_by_field(
            events, particle_counts, field_to_slice_by
        )
