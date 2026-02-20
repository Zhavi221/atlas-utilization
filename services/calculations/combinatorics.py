"""
Combinatorics for generating particle-type combinations.
"""
import itertools
from typing import Dict, List, Iterator


def make_objects_categories(
    object_types: List[str],
    min_n: int = 2,
    max_n: int = 4
) -> List[Dict[str, int]]:
    """
    Enumerate all event categories where each object type
    has between min_n and max_n entries.
    """
    limits = [range(min_n, max_n + 1) for _ in object_types]
    categories = []
    for counts in itertools.product(*limits):
        categories.append(dict(zip(object_types, counts)))
    return categories


def make_objects_combinations_for_category(
    category: Dict[str, int],
    min_k: int = 2,
    max_k: int = 4
) -> Iterator[Dict[str, int]]:
    """
    For a given event category, yield combinations where each particle type
    has between min_k and max_k particles.
    """
    object_types = list(category.keys())

    count_ranges = []
    for obj in object_types:
        available = category[obj]
        upper = min(max_k, available)
        if upper >= min_k:
            count_ranges.append(range(min_k, upper + 1))
        else:
            count_ranges.append(range(0, 1))

    for counts in itertools.product(*count_ranges):
        result = {obj: count for obj, count in zip(object_types, counts) if count > 0}
        if result:
            yield result


def get_all_combinations(
    object_types: List[str],
    min_particles: int,
    max_particles: int,
    min_count: int,
    max_count: int,
    limit: int = None
) -> List[Dict[str, int]]:
    categories = make_objects_categories(object_types, min_particles, max_particles)
    all_combinations = []
    seen = set()

    for category in categories:
        for combo in make_objects_combinations_for_category(category, min_count, max_count):
            combo_key = frozenset(combo.items())
            if combo_key not in seen:
                seen.add(combo_key)
                all_combinations.append(combo)

    return all_combinations[:limit]
