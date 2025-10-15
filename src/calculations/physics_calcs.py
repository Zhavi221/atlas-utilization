import awkward as ak
import numpy as np
import vector

PARTICLE_MASSES = {
    'Muons': 0.10566,
    'Electrons': 0.000511,
    'Tau': 1.777,
    'Jets': 0.0,
    'Photons': 0.0
    # Add other particles as needed
}

#TODO: check if this works, prettify
def calc_inv_mass_v2(particle_events: ak.Array) -> ak.Array:
    if len(particle_events) == 0:
        return ak.Array([])

    all_vectors = concat_events(particle_events)
    combined_vectors = ak.concatenate(all_vectors, axis=1)

    total_momentum = ak.sum(combined_vectors, axis=1)
    
    inv_mass = total_momentum.mass

    return inv_mass

def concat_events(particle_events: ak.Array) -> ak.Array:
    all_vectors = []
    particle_types = particle_events.fields
    for particle_type in particle_types:
        particle_array = particle_events[particle_type]

        #TODO: make this more robust
        mass = get_particle_mass(particle_type, particle_array)
        
        momentum_vector = vector.zip({
            "rho": particle_array.rho,
            "phi": particle_array.phi,
            "eta": particle_array.eta,
            "mass": mass  
        })
        
        all_vectors.append(momentum_vector)

    return all_vectors 

def get_particle_mass(particle_type, particle_array):
    # tau can refer to mass/energy
    if 'tau' in particle_array.fields:
        return particle_array.tau
    elif 'm' in particle_array.fields:
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

def filter_events_by_particle_counts(events, particle_counts, is_particle_counts_range=True):
    """
    MEMORY OPTIMIZED VERSION: Builds combined mask first, applies once.
    This avoids creating intermediate filtered arrays.
    """
    if len(events) == 0:
        return events
    
    # Start with all events passing (True mask)
    combined_mask = ak.ones_like(ak.num(events[events.fields[0]]), dtype=bool)
    
    for obj, range_dict in particle_counts.items():
        actual_field = find_actual_field_name(events.fields, obj)
        if not actual_field:
            continue
        
        obj_array = events[actual_field]
        if ak.all(ak.is_none(obj_array)):
            continue
        
        # Count particles
        if "Vector" in str(obj_array.type) or "Momentum" in str(obj_array.type):
            obj_count = ak.num(obj_array)
        else:
            obj_count = obj_array
        
        # Build mask for this particle type
        if not is_particle_counts_range:
            particle_mask = (obj_count == range_dict)
        else:
            particle_mask = (obj_count >= range_dict['min']) & (obj_count <= range_dict['max'])
        
        # Combine with overall mask using logical AND
        combined_mask = combined_mask & particle_mask
    
    # Apply filter only ONCE at the end
    filtered_events = events[combined_mask]
    
    # Clean up intermediate variables
    del combined_mask
    gc.collect()
    
    return ak.to_packed(filtered_events)

def find_actual_field_name(fields, obj_name):
    '''
        Finds the actual field name in the events for a given object name.
        This is useful when the object name is a substring of the actual field name.
    '''
    for field in fields:
        if obj_name in field:
            return field
    return None