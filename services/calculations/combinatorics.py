"""
Combinatorics for generating particle-type combinations.

Each combination is a dict mapping particle type → (count, start_index).
For backward compatibility, plain int values are also accepted everywhere
and are treated as (count, start_index=0).

Examples
--------
Leading-only (include_subleading=False, default):
    {"Electrons": (1, 0), "Jets": (1, 0)}   → e0 + j0

With sub-leading (include_subleading=True):
    {"Electrons": (1, 0), "Jets": (1, 0)}   → e0 + j0
    {"Electrons": (1, 1), "Jets": (1, 0)}   → e1 + j0   (sub-leading e)
    {"Electrons": (1, 0), "Jets": (1, 1)}   → e0 + j1   (sub-leading j)
    {"Electrons": (1, 1), "Jets": (1, 1)}   → e1 + j1

Helper accessors (safe for both old int and new tuple values):
    get_count(v)  → number of particles to take
    get_start(v)  → starting rank index (0 = leading)
"""
import itertools
from typing import Dict, List, Tuple, Union

# Type alias: value in a combination dict is either a plain int (legacy)
# or a (count, start_index) tuple.
CombValue = Union[int, Tuple[int, int]]


def get_count(value: CombValue) -> int:
    """Return particle count from a combination value."""
    return value[0] if isinstance(value, tuple) else value


def get_start(value: CombValue) -> int:
    """Return start rank index from a combination value (0 = leading)."""
    return value[1] if isinstance(value, tuple) else 0


def get_all_combinations(
    object_types: List[str],
    min_particles: int,
    max_particles: int,
    min_count: int,
    max_count: int,
    max_total_particles: int = 4,
    limit: int = None,
    include_subleading: bool = False,
    max_subleading_index: int = 1,
) -> List[Dict[str, CombValue]]:
    """
    Generate all unique particle combinations.

    Args:
        object_types:         Particle type names (e.g. ["Electrons", "Muons", ...])
        min_particles:        Minimum number of particle *types* in a combination.
        max_particles:        Maximum number of particle *types* in a combination.
        min_count:            Minimum particle count per included type.
        max_count:            Maximum particle count per included type.
        max_total_particles:  Hard cap on total particles across all types.
        limit:                Optional cap on total combinations returned.
        include_subleading:   If True, also generate start-index variants so that
                              sub-leading particles (e₁, j₁, …) are included as
                              separate signatures.
        max_subleading_index: Highest start index to consider when
                              include_subleading=True.  Default 1 means we go up
                              to the first sub-leading particle (index 1).
                              Increase to 2 for sub-sub-leading, etc.

    Returns:
        List of combination dicts.  Each value is a (count, start_index) tuple.
        start_index=0 for all leading-only combinations.
    """
    base_combinations: List[Dict[str, CombValue]] = []
    seen: set = set()

    n_types = min(max_particles, len(object_types))
    for r in range(min_particles, n_types + 1):
        for chosen_types in itertools.combinations(object_types, r):
            count_ranges = [range(min_count, max_count + 1) for _ in chosen_types]
            for counts in itertools.product(*count_ranges):
                if sum(counts) > max_total_particles:   # skip combinations exceeding particle cap
                    continue
                if sum(counts) < 2:                     # skip single-particle IM (physically meaningless)
                    continue
                # Store as (count, start=0) tuples
                combo = {ptype: (cnt, 0) for ptype, cnt in zip(chosen_types, counts)}
                combo_key = frozenset((k, v) for k, v in combo.items())
                if combo_key not in seen:
                    seen.add(combo_key)
                    base_combinations.append(combo)

    if not include_subleading:
        result = base_combinations
    else:
        # Pass an empty seen set so _expand_with_subleading always includes
        # the all-zero (leading) variants alongside the sub-leading ones.
        result = base_combinations + _expand_with_subleading(
            base_combinations, max_subleading_index, seen
        )

    if limit is not None:
        return result[:limit]
    return result


def _expand_with_subleading(
    base_combinations: List[Dict[str, CombValue]],
    max_subleading_index: int,
    seen: set,
) -> List[Dict[str, CombValue]]:
    """
    For each base combination, generate all variants where each particle
    type's start index is independently varied from 0 to max_subleading_index,
    subject to the constraint that start_index + count ≤ max_subleading_index + 1
    (i.e. we don't ask for particles beyond what we expect to exist).

    The original (all-zero start) combination is always first.
    """
    all_combinations: List[Dict[str, CombValue]] = []

    for combo in base_combinations:
        particle_types = list(combo.keys())

        # Build list of possible (count, start) values per particle type
        per_type_variants = []
        for ptype in particle_types:
            count, _ = combo[ptype]
            # start can range from 0 up to max_subleading_index,
            # but only if start + count - 1 <= max_subleading_index
            max_start = max_subleading_index - count + 1
            if max_start < 0:
                max_start = 0
            variants = [(count, s) for s in range(0, max_start + 1)]
            per_type_variants.append(variants)

        # Cartesian product of per-type start choices
        for value_combo in itertools.product(*per_type_variants):
            new_combo = dict(zip(particle_types, value_combo))
            combo_key = frozenset((k, v) for k, v in new_combo.items())
            if combo_key not in seen:
                seen.add(combo_key)
                all_combinations.append(new_combo)

    return all_combinations