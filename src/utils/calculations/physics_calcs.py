"""
Physics calculations for particle event processing.

Provides functions for:
- Invariant mass calculations
- Event filtering by kinematics and particle counts
- Final state grouping
- Event slicing and manipulation
"""
import awkward as ak
import numpy as np
import vector
import gc
from typing import Dict, Iterator, Tuple, Optional
from src.utils.calculations import consts

# vector.register_awkward()

def calc_inv_mass(particle_events: ak.Array) -> ak.Array:
    """
    Calculate invariant mass for particle events.
    
    Args:
        particle_events: Awkward array with particle fields (Electrons, Muons, etc.)
        
    Returns:
        Array of invariant masses (one per event)
    """
    if len(particle_events) == 0:
        return ak.Array([])

    all_vectors = concat_events(particle_events)
    combined_vectors = ak.concatenate(all_vectors, axis=1)

    total_momentum = ak.sum(combined_vectors, axis=1)
    
    if hasattr(total_momentum, 'tau'):
        inv_mass = total_momentum.tau
    else:
        inv_mass = total_momentum.mass

    return inv_mass

def concat_events(particle_events: ak.Array) -> list:
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

        mass = get_particle_known_mass(particle_type, particle_array)
        
        momentum_vector = vector.zip({
            "pt": particle_array.pt,
            "phi": particle_array.phi,
            "eta": particle_array.eta,
            "mass": mass  
        })
        
        all_vectors.append(momentum_vector)

    return all_vectors 

def get_particle_known_mass(particle_type: str, particle_array: ak.Array) -> ak.Array:
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

def extract_object_types(fields: list) -> set:
    particle_types = set()
    for field in fields:
        if '_' in field and not field.startswith('n'):
            particle_type = field.split('_')[0]
            particle_types.add(particle_type)

    return particle_types

def group_by_final_state(events: ak.Array) -> Iterator[Tuple[str, ak.Array]]:
    """
    Group events by their final state (particle counts).
    
    Args:
        events: Awkward array with particle events
        
    Yields:
        Tuples of (final_state_string, events_matching_final_state)
        Final state format: "{e}e_{m}m_{j}j_{g}g" where numbers are counts
        
    Note:
        If a particle type doesn't exist in the events, its count defaults to 0.
    """
    particle_counts = ak.num(events)
    
    # Safely get particle counts, defaulting to 0 if particle type doesn't exist
    # This handles cases where certain particle types aren't present in the data
    num_events = len(events)
    zero_array = ak.Array([0] * num_events) if num_events > 0 else ak.Array([])
    
    e = getattr(particle_counts, "Electrons", zero_array)
    m = getattr(particle_counts, "Muons", zero_array)
    j = getattr(particle_counts, "Jets", zero_array)
    g = getattr(particle_counts, "Photons", zero_array)
    
    all_events_fs = [f"{e}e_{m}m_{j}j_{g}g" for e, m, j, g in zip(e, m, j, g)]

    unique_fs = set(all_events_fs)
    
    for fs in unique_fs:
        mask = (ak.Array(all_events_fs) == fs)
        events_matching_fs = events[mask]
        fs = limit_particles_in_fs(fs, 4)
        yield (fs, events_matching_fs)

def limit_particles_in_fs(final_state, threshold):
    """
    Limit particle counts in final state string to threshold.
    
    Args:
        final_state: Final state string like "2e_3m_5j_1g"
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

def is_finalstate_contain_combination(final_state, combination):
    """
    Check if a final state contains enough particles for a combination.
    
    Args:
        final_state: Final state string like "2e_3m_5j_1g"
        combination: Dictionary mapping particle types to required counts
        
    Returns:
        True if final state has enough particles for the combination
    """
    fs_particles = final_state.split('_')
    for str_amount_particle in fs_particles:
        if len(str_amount_particle) < 2:
            continue
        amount_to_calc = str_amount_particle[0]
        particle_letter = str_amount_particle[1]
        particle = consts.LETTER_PARTICLE_MAPPING.get(particle_letter)
        
        if particle is None or particle not in combination:
            continue
        
        if not amount_to_calc.isdigit():
            continue
        
        fs_particle_amount = int(amount_to_calc)
        required_amount = combination[particle]
        
        if fs_particle_amount < required_amount:
            return False
        
    return True

def filter_events_by_particle_counts(
    events: ak.Array, 
    particle_counts: Dict[str, int], 
    is_exact_count: bool = False, 
    is_particle_counts_range: bool = False
) -> ak.Array:
    """
    Filter events by particle counts.
    
    Memory optimized: builds combined mask first, applies once.
    This avoids creating intermediate filtered arrays.
    
    When is_exact_count=True (used for combinations), also extracts only
    the specified particle types to ensure invariant mass calculations
    only include the combination particles.
    
    Args:
        events: Events to filter
        particle_counts: Dictionary of particle type to count/range
        is_exact_count: If True, require exact count match and extract only specified particle types
        is_particle_counts_range: If True, particle_counts contains min/max dicts
        
    Returns:
        Filtered events array (with only specified particle types if is_exact_count=True)
    """

    if len(events) == 0:
        return events
    
    # Start with all events passing (True mask)
    combined_mask = ak.ones_like(ak.num(events[events.fields[0]]), dtype=bool)
    
    for obj, value in particle_counts.items():
        actual_field = find_actual_field_name(events.fields, obj)
        if not actual_field:
            continue
        
        obj_array = events[actual_field]
        if ak.all(ak.is_none(obj_array)):
            continue
        
        # Count particles
        obj_count = ak.num(obj_array)
        
        # Build mask for this particle type
        if is_particle_counts_range:
            range_dict = value
            particle_mask = (obj_count >= range_dict['min']) & (obj_count <= range_dict['max'])
        elif is_exact_count:
            count = value
            particle_mask = (obj_count == count)
        else:
            count = value
            particle_mask = (obj_count >= count)
        
        combined_mask = combined_mask & particle_mask
    
    
    # Apply filter only ONCE at the end
    filtered_events = events[combined_mask]

    del combined_mask
    gc.collect()
    
    # If filtering by exact count (combinations), extract only specified particle types
    # This ensures invariant mass calculations only include the combination particles
    if is_exact_count:
        # Find actual field names for particle types in the combination
        fields_to_keep = {}
        for particle_type in particle_counts.keys():
            actual_field = find_actual_field_name(filtered_events.fields, particle_type)
            if actual_field:
                fields_to_keep[actual_field] = filtered_events[actual_field]
        
        # Create new array with only the specified fields
        if len(fields_to_keep) == 0:
            return ak.Array([])
        
        filtered_events = ak.zip(fields_to_keep, depth_limit=1)
    
    return ak.to_packed(filtered_events)

def slice_events_by_field(
    events: ak.Array, 
    particle_counts: Dict[str, int], 
    field_to_slice_by: str
) -> ak.Array:
    """
    Slice events to keep only top N particles by a field (e.g., pt).
    
    Args:
        events: Events to slice (should already contain only desired particle types)
        particle_counts: Dictionary mapping particle types to counts to keep
        field_to_slice_by: Field name to sort by (default: "pt")
        
    Returns:
        Sliced events array with top N particles of each type
    """
    for obj, count in particle_counts.items():
        actual_field = find_actual_field_name(events.fields, obj)
        if not actual_field:
            continue
        
        obj_array = events[actual_field]
        sorted_obj_array = obj_array[ak.argsort(obj_array[field_to_slice_by], ascending=False)]
        sliced_obj_array = sorted_obj_array[:, :count]
        events[actual_field] = sliced_obj_array

    return events

def filter_events_by_kinematics(
    events: ak.Array, 
    kinematic_cuts: Dict[str, Dict[str, float]]
) -> ak.Array:
    """
    Filter events by kinematic cuts (pt, eta, phi).
    
    Memory optimized: builds masks per particle type, applies once per type.
    Avoids creating multiple intermediate filtered arrays.
    
    Args:
        events: Events to filter
        kinematic_cuts: Dictionary with keys like "pt", "eta", "phi"
                      containing min/max values
                      
    Returns:
        Filtered events array
    """
    filtered_events = {}
    i = 0
    for obj in events.fields:
        particles = events[obj]

        # Skip empty arrays
        if len(particles.fields) == 0:
            filtered_events[obj] = particles
            continue

        mask_by = None
        if hasattr(particles, "pt"): 
            mask_by = particles.pt 
        mask = ak.ones_like(mask_by, dtype=bool)

        if "pt" in kinematic_cuts and hasattr(particles, "pt"):
            pt_vals = ak.values_astype(particles.pt, float)
            mask = mask & (pt_vals >= kinematic_cuts["pt"]["min"])

        if "eta" in kinematic_cuts and hasattr(particles, "eta"):
            eta_vals = ak.values_astype(particles.eta, float)
            mask = mask & (eta_vals >= kinematic_cuts["eta"]["min"]) & (eta_vals <= kinematic_cuts["eta"]["max"])

        if "phi" in kinematic_cuts and hasattr(particles, "phi"):
            phi_vals = ak.values_astype(particles.phi, float)
            mask = mask & (phi_vals >= kinematic_cuts["phi"]["min"]) & (phi_vals <= kinematic_cuts["phi"]["max"])

        # Apply mask once
        filtered_events[obj] = ak.mask(particles, mask)

    return ak.zip(filtered_events, depth_limit=1)

def find_actual_field_name(fields: list, obj_name: str) -> Optional[str]:
    """
    Find the actual field name in the events for a given object name.
    
    This is useful when the object name is a substring of the actual field name.
    
    Args:
        fields: List of available field names
        obj_name: Object name to search for
        
    Returns:
        Matching field name or None if not found
    """
    for field in fields:
        if obj_name in field:
            return field
    return None