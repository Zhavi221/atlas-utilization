import awkward as ak
import numpy as np
import vector
import gc

vector.register_awkward()

PARTICLE_MASSES = {
    'Muons': 0.10566,
    'Electrons': 0.000511,
    'Tau': 1.777,
    'Jets': 0.0,
    'Photons': 0.0
}

def calc_inv_mass(particle_events: ak.Array, combination: dict={}, by_highest_pt=False) -> ak.Array:
    if len(particle_events) == 0:
        return ak.Array([])

    if by_highest_pt:
        #TODO sort by pT and slice by counts from combination
        pass
    all_vectors = concat_events(particle_events)
    combined_vectors = ak.concatenate(all_vectors, axis=1)

    total_momentum = ak.sum(combined_vectors, axis=1)
    
    if hasattr(total_momentum, 'tau'):
        inv_mass = total_momentum.tau
    else:
        inv_mass = total_momentum.mass

    return inv_mass

def concat_events(particle_events: ak.Array) -> ak.Array:
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

def get_particle_known_mass(particle_type, particle_array):
    if 'm' in particle_array.fields:
        return particle_array.m
    else:
        return PARTICLE_MASSES.get(particle_type, 0.0) 

def extract_object_types(fields: list) -> set:
    particle_types = set()
    for field in fields:
        if '_' in field and not field.startswith('n'):
            particle_type = field.split('_')[0]
            particle_types.add(particle_type)

    return particle_types

def group_by_final_state(events: ak.Array) -> ak.Array:
    particle_counts = ak.num(events)
    e = particle_counts.Electrons
    m = particle_counts.Muons
    j = particle_counts.Jets
    p = particle_counts.Photons
    all_events_fs = [f"{e}{m}{j}{p}" for e, m, j, p in zip(e, m, j, p)]

    unique_fs = set(all_events_fs)
    
    for fs in unique_fs:
        mask = (ak.Array(all_events_fs) == fs)
        events_matching_fs = events[mask]
        yield events_matching_fs

def filter_events_by_particle_counts(events, particle_counts, is_exact_count=False, is_particle_counts_range=False):
    """
    MEMORY OPTIMIZED VERSION: Builds combined mask first, applies once.
    This avoids creating intermediate filtered arrays.
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
        
        # Combine with overall mask using logical AND
        combined_mask = combined_mask & particle_mask
    
    # Apply filter only ONCE at the end
    filtered_events = events[combined_mask]
    
    # Clean up intermediate variables
    del combined_mask
    gc.collect()
    
    return ak.to_packed(filtered_events)

def filter_events_by_kinematics(events, kinematic_cuts):
    """
    MEMORY OPTIMIZED: Build masks per particle type, apply once per type.
    Avoids creating multiple intermediate filtered arrays.
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

def find_actual_field_name(fields, obj_name):
    '''
        Finds the actual field name in the events for a given object name.
        This is useful when the object name is a substring of the actual field name.
    '''
    for field in fields:
        if obj_name in field:
            return field
    return None