import awkward as ak
import numpy as np
import vector
import gc

PARTICLE_MASSES = {
    'Muons': 0.10566,
    'Electrons': 0.000511,
    'Tau': 1.777,
    'Jets': 0.0,
    'Photons': 0.0
    # Add other particles as needed
}

def calc_inv_mass(particle_events: ak.Array) -> ak.Array:
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

def filter_events_by_kinematics(events, kinematic_cuts):
    """
    MEMORY OPTIMIZED: Build masks per particle type, apply once per type.
    Avoids creating multiple intermediate filtered arrays.
    """
    filtered_events = {}
    i = 0
    print(len(events.fields))
    for obj in events.fields:
        print('first')
        particles = events[obj]

        # Skip empty arrays
        if len(particles) == 0:
            filtered_events[obj] = particles
            print('thats it')
            continue

        # Start with all True mask
        mask = ak.ones_like(particles.rho if hasattr(particles, "rho") else particles.pt, dtype=bool)
        print('second')

        # Apply all cuts to the same mask (no intermediate arrays)
        if "rho" in kinematic_cuts and hasattr(particles, "rho"):
            rho_vals = ak.values_astype(particles.rho, float)
            mask = mask & (rho_vals >= kinematic_cuts["rho"]["min"])

        if "eta" in kinematic_cuts and hasattr(particles, "eta"):
            eta_vals = ak.values_astype(particles.eta, float)
            mask = mask & (eta_vals >= kinematic_cuts["eta"]["min"]) & (eta_vals <= kinematic_cuts["eta"]["max"])

        if "phi" in kinematic_cuts and hasattr(particles, "phi"):
            phi_vals = ak.values_astype(particles.phi, float)
            mask = mask & (phi_vals >= kinematic_cuts["phi"]["min"]) & (phi_vals <= kinematic_cuts["phi"]["max"])

        if "tau" in kinematic_cuts and hasattr(particles, "tau"):
            tau_vals = ak.values_astype(particles.tau, float)
            mask = mask & (tau_vals >= kinematic_cuts["tau"]["min"])

        # Apply mask once
        print('third')
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