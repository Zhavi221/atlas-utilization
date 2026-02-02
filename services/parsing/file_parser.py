"""
FileParser service - Single responsibility: Parse ROOT files.

Extracts events from ATLAS ROOT files using uproot.
No orchestration logic, no state management.
"""

import logging
import awkward as ak
import uproot
import itertools
from typing import Optional

from src.ParseAtlas import schemas


class FileParser:
    """
    Service for parsing individual ROOT files.
    
    Pure function-like service with no state. All methods are static.
    """
    
    @staticmethod
    def parse_file(
        file_path: str,
        tree_names: list[str],
        release_year: str,
        batch_size: int = 40_000
    ) -> Optional[ak.Array]:
        """
        Parse a single ROOT file and return events.
        
        Args:
            file_path: Path or URI to ROOT file
            tree_names: List of possible tree names to search for
            release_year: Release year identifier (e.g., "2024r-pp")
            batch_size: Number of entries to process per batch
            
        Returns:
            Awkward array of events with particle objects, or None if parsing failed
        """
        try:
            with uproot.open(file_path) as root_file:
                return FileParser._parse_opened_file(
                    root_file,
                    tree_names,
                    release_year,
                    batch_size,
                    file_path
                )
        except Exception as e:
            logging.warning(f"Failed to open file {file_path}: {e}")
            return None
    
    @staticmethod
    def _parse_opened_file(
        root_file,
        tree_names: list[str],
        release_year: str,
        batch_size: int,
        file_path: str
    ) -> Optional[ak.Array]:
        """Parse an already-opened ROOT file."""
        # 1. Find the data tree
        tree_name = FileParser._get_data_tree_name(root_file.keys(), tree_names)
        tree = root_file[tree_name]
        all_tree_branches = set(tree.keys())
        n_entries = tree.num_entries
        
        # 2. Extract object branches based on schema
        obj_branches = FileParser._extract_branches_by_schema(
            all_tree_branches,
            release_year
        )
        
        if not obj_branches:
            logging.warning(f"No particles found in schema for file {file_path}")
            return None
        
        # 3. Test branch accessibility
        obj_branches = FileParser._filter_accessible_branches(tree, obj_branches)
        
        if not obj_branches:
            logging.warning(f"No accessible particles found in file {file_path}")
            return None
        
        # 4. Read file in batches
        all_branches = set(itertools.chain.from_iterable(obj_branches.values()))
        obj_events = FileParser._read_file_in_batches(
            tree,
            all_branches,
            obj_branches,
            n_entries,
            batch_size
        )
        
        # 5. Zip into final structure
        return ak.zip(obj_events, depth_limit=1)
    
    @staticmethod
    def _get_data_tree_name(
        root_file_keys: list[str],
        possible_tree_names: list[str]
    ) -> str:
        """
        Find the data tree name in the ROOT file.
        
        Args:
            root_file_keys: All keys in the ROOT file
            possible_tree_names: List of possible tree names to check
            
        Returns:
            Tree name to use (defaults to "CollectionTree" if none found)
        """
        if not possible_tree_names:
            return "CollectionTree"
        
        # Remove trailing ';1' from ROOT keys
        available_trees = [key[:-2] if key.endswith(';1') else key for key in root_file_keys]
        
        for tree_name in possible_tree_names:
            if tree_name in available_trees:
                return tree_name
        
        # Fallback
        return "CollectionTree"
    
    @staticmethod
    def _extract_branches_by_schema(
        tree_branches: set[str],
        release_year: str
    ) -> dict[str, dict[str, str]]:
        """
        Extract branches by object based on release-specific schema.
        
        Args:
            tree_branches: Set of all branch names in the ROOT tree
            release_year: Release year identifier
            
        Returns:
            Dict mapping object names to their branch mappings
            Format: {obj_name: {full_branch: quantity, ...}}
        """
        try:
            # Extract record_id from release_year if it's in "record_*" format
            record_id = None
            if release_year.startswith("record_"):
                try:
                    record_id = int(release_year.split("_")[1])
                except (ValueError, IndexError):
                    pass
            
            schema_config = schemas.get_schema_for_release(release_year, record_id=record_id)
        except KeyError:
            logging.warning(
                f"Release year '{release_year}' not found in schemas. "
                "Attempting auto-detection."
            )
            return FileParser._auto_detect_branches(tree_branches)
        
        obj_branches = {}
        objects = schema_config["objects"]
        naming_pattern = schema_config.get("naming_pattern", "dotted")
        
        for obj_name, fields in objects.items():
            if naming_pattern == "flat":
                obj_branches_for_obj = FileParser._extract_flat_branches(
                    obj_name, fields, tree_branches, release_year
                )
            else:
                obj_branches_for_obj = FileParser._extract_dotted_branches(
                    obj_name, fields, tree_branches, release_year, schema_config
                )
            
            if obj_branches_for_obj:
                obj_branches[obj_name] = obj_branches_for_obj
        
        return obj_branches
    
    @staticmethod
    def _extract_flat_branches(
        obj_name: str,
        fields: list[str],
        tree_branches: set[str],
        release_year: str
    ) -> dict[str, str]:
        """Extract branches using flat naming pattern (object_field)."""
        # Import here to avoid circular dependency
        from src.ParseAtlas.parser import AtlasOpenDataChunkParser
        
        branch_base = AtlasOpenDataChunkParser._prepare_obj_branch_name(obj_name, release_year=release_year)
        available_fields = [
            f for f in fields if f"{branch_base}_{f}" in tree_branches
        ]
        
        if FileParser._can_calculate_inv_mass(available_fields):
            return {
                f"{branch_base}_{field}": field
                for field in available_fields
            }
        
        return {}
    
    @staticmethod
    def _extract_dotted_branches(
        obj_name: str,
        fields: list[str],
        tree_branches: set[str],
        release_year: str,
        schema_config: dict
    ) -> dict[str, str]:
        """Extract branches using dotted naming pattern (object.field)."""
        # Import here to avoid circular dependency
        from src.ParseAtlas.parser import AtlasOpenDataChunkParser
        
        branch_name = AtlasOpenDataChunkParser._prepare_obj_branch_name(obj_name, release_year=release_year)
        logging.info(f"ATLAS-style naming for {obj_name}, branch base: {branch_name}")
        
        # Check if schema has field_paths (CMS-style nested paths)
        field_paths = schema_config.get("field_paths", {})
        obj_field_paths = field_paths.get(obj_name, {})
        
        if obj_field_paths:
            # CMS-style: use nested field paths
            from src.ParseAtlas.parser import AtlasOpenDataChunkParser
            return AtlasOpenDataChunkParser._find_cms_branches(
                branch_name, fields, obj_field_paths, tree_branches
            )
        
        # ATLAS-style: simple dotted naming
        branch_to_quantity = {}
        available_fields = []
        
        for field in fields:
            # Try the field name as-is first
            branch_full = f"{branch_name}.{field}"
            if branch_full in tree_branches:
                available_fields.append(field)
                branch_to_quantity[branch_full] = field
            # If not found and field is "mass", try "m"
            elif field == "mass":
                mass_branch = f"{branch_name}.m"
                if mass_branch in tree_branches:
                    available_fields.append(field)
                    branch_to_quantity[mass_branch] = field
        
        if FileParser._can_calculate_inv_mass(available_fields):
            return branch_to_quantity
        
        return {}
    
    @staticmethod
    def _can_calculate_inv_mass(
        available_fields: list[str],
        ref_system: set[str] = {'phi', 'eta', 'pt'}
    ) -> bool:
        """
        Check if we have the minimum fields needed to calculate invariant mass.
        
        Args:
            available_fields: List of available field names
            ref_system: Required fields for invariant mass calculation
            
        Returns:
            True if all required fields are present
        """
        return ref_system.issubset(set(available_fields))
    
    @staticmethod
    def _filter_accessible_branches(
        tree,
        obj_branches: dict[str, dict[str, str]]
    ) -> dict[str, dict[str, str]]:
        """
        Test branch accessibility and filter out inaccessible ones.
        
        Args:
            tree: ROOT tree object
            obj_branches: Object to branch mappings
            
        Returns:
            Filtered object to branch mappings with only accessible branches
        """
        accessible_obj_branches = {}
        
        for obj_name, branch_mapping in obj_branches.items():
            accessible_branches = {}
            
            for branch_path, quantity in branch_mapping.items():
                try:
                    # Test read one entry
                    test_arr = tree.arrays(branch_path, entry_start=0, entry_stop=1, library="ak")
                    if branch_path in test_arr.fields:
                        accessible_branches[branch_path] = quantity
                except Exception:
                    # Branch doesn't exist or isn't accessible
                    continue
            
            if accessible_branches and FileParser._can_calculate_inv_mass(
                list(accessible_branches.values())
            ):
                accessible_obj_branches[obj_name] = accessible_branches
        
        return accessible_obj_branches
    
    @staticmethod
    def _read_file_in_batches(
        tree,
        all_branches: set[str],
        obj_branches: dict[str, dict[str, str]],
        n_entries: int,
        batch_size: int
    ) -> dict[str, ak.Array]:
        """
        Read file in batches and organize by object.
        
        Args:
            tree: ROOT tree object
            all_branches: Set of all branch names to read
            obj_branches: Object to branch mappings
            n_entries: Total number of entries in tree
            batch_size: Number of entries per batch
            
        Returns:
            Dict mapping object names to zipped awkward arrays
        """
        # Initialize storage for batches
        obj_events_by_quantities = {
            obj_name: [] for obj_name in obj_branches.keys()
        }
        
        # Define entry ranges
        is_file_big = n_entries > batch_size
        if is_file_big:
            entry_ranges = [
                (start, min(start + batch_size, n_entries))
                for start in range(0, n_entries, batch_size)
            ]
        else:
            entry_ranges = [(0, n_entries)]
        
        # Read batches
        for entry_start, entry_stop in entry_ranges:
            try:
                batch_data = tree.arrays(
                    all_branches,
                    entry_start=entry_start,
                    entry_stop=entry_stop,
                    library="ak"
                )
            except Exception as e:
                logging.warning(f"Error reading batch {entry_start}-{entry_stop}: {e}")
                continue
            
            # Organize by object
            for obj_name, branch_mapping in obj_branches.items():
                available_branches = [
                    b for b in branch_mapping.keys() if b in batch_data.fields
                ]
                if available_branches:
                    subset = batch_data[available_branches]
                    if len(subset) > 0:
                        obj_events_by_quantities[obj_name].append(subset)
        
        # Concatenate batches and zip by quantity
        result = {}
        for obj_name, chunks in obj_events_by_quantities.items():
            if chunks:
                concatenated = ak.concatenate(chunks)
                result[obj_name] = ak.zip({
                    quantity: concatenated[full_branch]
                    for full_branch, quantity in obj_branches[obj_name].items()
                })
        
        return result
    
    @staticmethod
    def _auto_detect_branches(tree_branches: set[str]) -> dict[str, dict[str, str]]:
        """
        Auto-detect branch structure when schema is not available.
        
        Fallback method when release year is not in schemas.
        
        Args:
            tree_branches: Set of all branch names
            
        Returns:
            Best-guess object to branch mappings
        """
        # Import here to avoid circular dependency
        from src.ParseAtlas.parser import AtlasOpenDataChunkParser
        return AtlasOpenDataChunkParser._auto_detect_branches(tree_branches)
