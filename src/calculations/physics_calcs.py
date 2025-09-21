import awkward as ak
import numpy as np

#TODO: go over this and understand
def calculate_inv_mass(events: ak.Array) -> ak.Array:
        """
        Compute invariant mass per event from the combined objects.
        Compatible with flattened structure where each particle type has separate fields.
        Expected format: ParticleType_kinematic (e.g., 'Jets_rho', 'Jets_phi', etc.)
        """
        if len(events) == 0:
            return ak.Array([])

        # Extract unique particle types from field names (excluding count fields)
        particle_types = set()
        for field in events.fields:
            if '_' in field and not field.startswith('n'):
                particle_type = field.split('_')[0]
                particle_types.add(particle_type)


        num_events = len(events)
        total_px = np.zeros(num_events)
        total_py = np.zeros(num_events)
        total_pz = np.zeros(num_events)
        total_energy = np.zeros(num_events)

        for particle_type in particle_types:
            print(f"Processing {particle_type}...")
            
            # Check if we have the required kinematic fields
            rho_field = f"{particle_type}_rho"
            phi_field = f"{particle_type}_phi"
            eta_field = f"{particle_type}_eta"
            
            if not all(field in events.fields for field in [rho_field, phi_field, eta_field]):
                print(f"Missing required fields for {particle_type}")
                continue
                
            rho = events[rho_field]
            phi = events[phi_field] 
            eta = events[eta_field]
            
            # Convert rho to pt (assuming rho is transverse momentum)
            pt = rho
            
            # Assign mass - default masses for common particles (in GeV/c²)
            particle_key = particle_type.lower()
            default_masses = {
                'jets': 0.0,
                'photons': 0.0,
                'electrons': 0.000511,  # Convert from MeV to GeV
                'muons': 0.10566      # Convert from MeV to GeV
            }
            
            # Check for mass field first, then use defaults
            mass_field = f"{particle_type}_m"
            if mass_field in events.fields:
                mass = events[mass_field]
            elif particle_key in default_masses:
                mass = ak.full_like(pt, default_masses[particle_key])
            else:
                print(f"⚠️  Warning: No mass found for {particle_type}, assuming massless")
                mass = ak.full_like(pt, 0.0)

            # Handle the jagged structure - pt, eta, phi, mass should all be jagged arrays
            # Convert to rectangular components for each event
            
            # Calculate px, py, pz, energy for each particle
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)  
            pz = pt * np.sinh(eta)
            energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
            
            # Sum over particles in each event
            try:
                px_sum = ak.sum(px, axis=1, keepdims=False)
                py_sum = ak.sum(py, axis=1, keepdims=False)
                pz_sum = ak.sum(pz, axis=1, keepdims=False)
                energy_sum = ak.sum(energy, axis=1, keepdims=False)
                
                # Convert to numpy arrays and add to totals
                total_px += ak.to_numpy(px_sum)
                total_py += ak.to_numpy(py_sum)
                total_pz += ak.to_numpy(pz_sum)
                total_energy += ak.to_numpy(energy_sum)
                
            except Exception as e:
                print(f"Error processing {particle_type}: {e}")
                print(f"  px type: {ak.type(px)}")
                print(f"  Trying alternative approach...")
                
                # Alternative: handle each event individually
                for i in range(len(events)):
                    try:
                        if len(px[i]) > 0:
                            total_px[i] += np.sum(ak.to_numpy(px[i]))
                            total_py[i] += np.sum(ak.to_numpy(py[i]))
                            total_pz[i] += np.sum(ak.to_numpy(pz[i]))
                            total_energy[i] += np.sum(ak.to_numpy(energy[i]))
                    except:
                        continue

        # Calculate invariant mass from total 4-momentum
        total_mass_squared = total_energy**2 - (total_px**2 + total_py**2 + total_pz**2)
        invariant_mass = np.where(total_mass_squared >= 0, np.sqrt(total_mass_squared), 0)
        
        print(f"Calculated {len(invariant_mass)} invariant masses")
        print(f"Mass range: {np.min(invariant_mass)} to {np.max(invariant_mass)}")
        
        return ak.Array(invariant_mass)

def filter_events_by_combination(events, particle_counts, use_count_range=True):
    '''
        Filters the events by the given combination dictionary.
    '''
    for obj, range_dict in particle_counts.items():
        if not (obj := find_actual_field_name(events.fields, obj)):
            continue
        
        obj_array = events[obj]
        if ak.all(ak.is_none(obj_array)):
            continue

        # obj_count = ak.num(obj_array)
        obj_count = obj_array
        # Use bitwise & instead of logical and for array operations
        if not use_count_range:
            mask = obj_count == range_dict
        else:
            mask = (obj_count >= range_dict['min']) & (obj_count <= range_dict['max'])
        
        events = events[mask]

    return ak.to_packed(events)

def find_actual_field_name(fields, obj_name):
    '''
        Finds the actual field name in the events for a given object name.
        This is useful when the object name is a substring of the actual field name.
    '''
    for field in fields:
        if obj_name in field:
            return field
    return None