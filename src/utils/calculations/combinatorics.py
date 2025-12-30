import itertools
from typing import Dict, List, Iterator, Tuple
import json

def make_objects_categories(
    object_types: List[str],
    min_n: int = 2,
    max_n: int = 4
) -> List[Dict[str, int]]:
    """
    Enumerate all event categories where each object type 
                    has between min_n and max_n entries.
    """
    limits = [range(min_n, max_n+1) for _ in object_types]
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
    
    Output format: {'Electrons': 2, 'Muons': 3} means 2 electrons and 3 muons.
    Each count is guaranteed to be in range [min_k, max_k].
    """
    object_types = list(category.keys())
    
    # For each type, the count range is [min_k, min(max_k, available)]
    count_ranges = []
    for obj in object_types:
        available = category[obj]
        upper = min(max_k, available)
        if upper >= min_k:
            count_ranges.append(range(min_k, upper + 1))
        else:
            # This type doesn't have enough particles, skip it
            count_ranges.append(range(0, 1))  # Only 0 (exclude this type)
    
    for counts in itertools.product(*count_ranges):
        # Only include types with count > 0
        result = {obj: count for obj, count in zip(object_types, counts) if count > 0}
        if result:  # Don't yield empty combinations
            yield result
            
def get_all_combinations(object_types, min_particles, max_particles, min_count, max_count, limit=None):
    categories = make_objects_categories(object_types, min_particles, max_particles)
    all_combinations = []
    seen = set()
    
    for category in categories:
        category_combinations = list(make_objects_combinations_for_category(category, min_count, max_count))
        for combo in category_combinations:
            # Convert dict to frozenset for hashing to deduplicate
            combo_key = frozenset(combo.items())
            if combo_key not in seen:
                seen.add(combo_key)
                all_combinations.append(combo)
    
    return all_combinations[:limit]
    # return categories[:limit]

if __name__ == "__main__":
    all_combinations = get_all_combinations(
        ['Electrons', 'Muons', 'Jets', 'Photons'])
    print(all_combinations)
    print(f"Generated {len(all_combinations)} combinations.\n")
