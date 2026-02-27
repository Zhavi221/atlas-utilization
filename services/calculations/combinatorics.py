"""
Combinatorics for generating particle-type combinations.
"""
import itertools
from typing import Dict, List, Iterator


def get_all_combinations(
    object_types: List[str],
    min_particles: int,
    max_particles: int,
    min_count: int,
    max_count: int,
    limit: int = None
) -> List[Dict[str, int]]:
    """
    Generate all unique particle combinations.

    Args:
        object_types: particle type names (e.g. ["Electrons", "Muons", ...])
        min_particles: minimum number of particle *types* in a combination
        max_particles: maximum number of particle *types* in a combination
        min_count: minimum count per included type
        max_count: maximum count per included type
    """
    all_combinations: List[Dict[str, int]] = []
    seen: set = set()

    n_types = min(max_particles, len(object_types))
    for r in range(min_particles, n_types + 1):
        for chosen_types in itertools.combinations(object_types, r):
            count_ranges = [range(min_count, max_count + 1) for _ in chosen_types]
            for counts in itertools.product(*count_ranges):
                combo = dict(zip(chosen_types, counts))
                combo_key = frozenset(combo.items())
                if combo_key not in seen:
                    seen.add(combo_key)
                    all_combinations.append(combo)

    if limit is not None:
        return all_combinations[:limit]
    return all_combinations
