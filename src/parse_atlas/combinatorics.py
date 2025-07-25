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
) -> Iterator[Tuple[List[str], Tuple[str, ...]]]:
    """
    For a given event category and K in min_k...max_k (number of object types),
    yield (selected object types, tuple of object labels) combinations.
    """
    available_types = [obj for obj, n in category.items() if n > 0]
    max_k = min(max_k, len(available_types))
    if max_k < min_k:
        return
    
    min_k = min_k - 1
    for k in range(min_k, max_k+1):
        type_combos = itertools.combinations(category, k)
        type_combos = [list(combo_tuple) for combo_tuple in type_combos]
        
        category['Jets'] = k
        for type_combo in type_combos:
            if 'Jets' not in type_combo:
                type_combo.append('Jets')
                
            label_pools = []
            for obj in type_combo:
                label_pools.append([f"{obj}{i+1}" for i in range(category[obj])])
            
            # Product across labels
            for obj_selection in itertools.product(*label_pools):
                obj_selection = {
                    obj[:-1]: int(obj[-1]) for obj in obj_selection}
                
                yield obj_selection
            
# --- Example usage ---
if __name__ == "__main__":
    object_types = ["Electrons", "Muons", "Photons"]
    cats = make_objects_categories(object_types, min_n=2, max_n=4)
    print(f"Generated {len(cats)} categories.\n")

    # Show first example category
    example_cat = cats[0]
    print("Example category:", example_cat)
    print(cats)
    combos = list(make_objects_combinations_for_category(example_cat, min_k=2, max_k=4))
    # print(json.dumps(combos[:3]))
    # print(f"Number of combinations: {len(combos)}")
    # for tcombo, labels in combos[:5]:
    #     print(f"  Types={tcombo}, Labels={labels}")
