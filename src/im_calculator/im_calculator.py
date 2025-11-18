import awkward as ak
import vector

from src.im_calculator import consts

class IM_Calculator():
    def __init__(self):
        pass

    def group_by_final_state(self):
        particle_counts = ak.num(self.particle_quantities_events)
        e = particle_counts.Electrons
        m = particle_counts.Muons
        j = particle_counts.Jets
        p = particle_counts.Photons
        all_events_fs = [f"{e}e_{m}m_{j}j_{p}p" for e, m, j, p in zip(e, m, j, p)]

        unique_fs = set(all_events_fs)
        
        for fs in unique_fs:
            mask = (ak.Array(all_events_fs) == fs)
            events_matching_fs = self.particle_quantities_events[mask]
            yield (fs, events_matching_fs)

    def calc_inv_mass(self) -> ak.Array:
        if len(self.particle_quantities_events) == 0:
            return ak.Array([])

        all_vectors = self.concat_events()
        combined_vectors = ak.concatenate(all_vectors, axis=1)

        total_momentum = ak.sum(combined_vectors, axis=1)
        
        if hasattr(total_momentum, 'tau'):
            inv_mass = total_momentum.tau
        else:
            inv_mass = total_momentum.mass

        return inv_mass

    def concat_events(self) -> ak.Array:
        all_vectors = []
        particle_types = self.particle_quantities_events.fields
        for particle_type in particle_types:
            particle_array = self.particle_quantities_events[particle_type]

            mass = IM_Calculator.get_particle_known_mass(particle_type, particle_array)
            
            momentum_vector = vector.zip({
                "pt": particle_array.pt,
                "phi": particle_array.phi,
                "eta": particle_array.eta,
                "mass": mass  
            })
            
            all_vectors.append(momentum_vector)

        return all_vectors 

    @staticmethod
    def get_particle_known_mass(particle_type, particle_array):
        if 'm' in particle_array.fields:
            return particle_array.m
        else:
            return consts.KNOWN_MASSES.get(particle_type, 0.0) 

    @staticmethod
    def does_fs_contain_combination(final_state, combination):
        fs_particles = final_state.split('_')
        for str_amount_particle in fs_particles:
            amount_to_calc = str_amount_particle[0]
            particle_letter = str_amount_particle[1]
            particle = consts.LETTER_PARTICLE_MAPPING[particle_letter]

            if particle not in combination or not amount_to_calc.isdigit():
                continue
            
            fs_particle_amount = combination[particle]
            if int(amount_to_calc) > fs_particle_amount:
                return False
            
        return True