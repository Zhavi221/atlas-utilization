"""
Physics calculations for particle event processing.

Provides functions for invariant mass calculations, event filtering
by kinematics and particle counts, final state grouping, and event slicing.
"""
import awkward as ak
import numpy as np
import vector
import gc
from typing import Dict, Iterator, Tuple, Optional

from services.calculations import consts


def calc_inv_mass(particle_events: ak.Array) -> ak.Array:
    if len(particle_events) == 0:
        return ak.Array([])

    all_vectors = concat_events(particle_events)
    combined_vectors = ak.concatenate(all_vectors, axis=1)
    total_momentum = ak.sum(combined_vectors, axis=1)

    if hasattr(total_momentum, 'tau'):
        return total_momentum.tau
    return total_momentum.mass


def concat_events(particle_events: ak.Array) -> list:
    all_vectors = []
    for particle_type in particle_events.fields:
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
    if 'm' in particle_array.fields:
        return particle_array.m
    return consts.KNOWN_MASSES.get(particle_type, 0.0)


def extract_object_types(fields: list) -> set:
    particle_types = set()
    for field in fields:
        if '_' in field and not field.startswith('n'):
            particle_type = field.split('_')[0]
            particle_types.add(particle_type)
    return particle_types


def group_by_final_state(events: ak.Array) -> Iterator[Tuple[str, ak.Array]]:
    num_events = len(events)
    zero_array = ak.Array([0] * num_events) if num_events > 0 else ak.Array([])
    particle_counts = ak.num(events)

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


def limit_particles_in_fs(final_state: str, threshold: int) -> str:
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


def is_finalstate_contain_combination(final_state: str, combination: Dict[str, int]) -> bool:
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
    if len(events) == 0:
        return events

    combined_mask = ak.ones_like(ak.num(events[events.fields[0]]), dtype=bool)

    for obj, value in particle_counts.items():
        actual_field = find_actual_field_name(events.fields, obj)
        if not actual_field:
            continue

        obj_array = events[actual_field]
        if ak.all(ak.is_none(obj_array)):
            continue

        obj_count = ak.num(obj_array)

        if is_particle_counts_range:
            range_dict = value
            particle_mask = (obj_count >= range_dict['min']) & (obj_count <= range_dict['max'])
        elif is_exact_count:
            particle_mask = (obj_count == value)
        else:
            particle_mask = (obj_count >= value)

        combined_mask = combined_mask & particle_mask

    filtered_events = events[combined_mask]
    del combined_mask
    gc.collect()

    if is_exact_count:
        fields_to_keep = {}
        for particle_type in particle_counts.keys():
            actual_field = find_actual_field_name(filtered_events.fields, particle_type)
            if actual_field:
                fields_to_keep[actual_field] = filtered_events[actual_field]

        if len(fields_to_keep) == 0:
            return ak.Array([])
        filtered_events = ak.zip(fields_to_keep, depth_limit=1)

    return ak.to_packed(filtered_events)


def slice_events_by_field(
    events: ak.Array,
    particle_counts: Dict[str, int],
    field_to_slice_by: str
) -> ak.Array:
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
    filtered_events = {}
    for obj in events.fields:
        particles = events[obj]

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

        filtered_events[obj] = ak.mask(particles, mask)

    return ak.zip(filtered_events, depth_limit=1)


def find_actual_field_name(fields: list, obj_name: str) -> Optional[str]:
    for field in fields:
        if obj_name in field:
            return field
    return None
