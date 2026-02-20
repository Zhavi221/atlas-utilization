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

from services.parsing import schemas
from services import consts


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
        tree_name = FileParser._get_data_tree_name(root_file.keys(), tree_names)
        tree = root_file[tree_name]
        all_tree_branches = set(tree.keys())
        n_entries = tree.num_entries
        
        obj_branches = FileParser._extract_branches_by_schema(
            all_tree_branches,
            release_year
        )
        
        if not obj_branches:
            logging.warning(f"No particles found in schema for file {file_path}")
            return None
        
        obj_branches = FileParser._filter_accessible_branches(tree, obj_branches)
        
        if not obj_branches:
            logging.warning(f"No accessible particles found in file {file_path}")
            return None
        
        all_branches = set(itertools.chain.from_iterable(obj_branches.values()))
        obj_events = FileParser._read_file_in_batches(
            tree,
            all_branches,
            obj_branches,
            n_entries,
            batch_size
        )
        
        return ak.zip(obj_events, depth_limit=1)
    
    @staticmethod
    def _get_data_tree_name(
        root_file_keys: list[str],
        possible_tree_names: list[str]
    ) -> str:
        if not possible_tree_names:
            return "CollectionTree"
        
        available_trees = [key[:-2] if key.endswith(';1') else key for key in root_file_keys]
        
        for tree_name in possible_tree_names:
            if tree_name in available_trees:
                return tree_name
        
        return "CollectionTree"
    
    @staticmethod
    def _extract_branches_by_schema(
        tree_branches: set[str],
        release_year: str
    ) -> dict[str, dict[str, str]]:
        """
        Extract branches by object based on release-specific schema.
        
        Returns:
            Dict mapping object names to their branch mappings
            Format: {obj_name: {full_branch: quantity, ...}}
        """
        try:
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
    def _prepare_obj_branch_name(
        obj_name: str,
        release_year: str = "2024r-pp",
        field: str = None,
        record_id: int = None
    ) -> str:
        """
        Prepare object branch name using release-specific template.

        For flat naming with a field, returns "ObjectName_field".
        For flat naming without a field, returns just the mapped object name.
        For dotted naming, returns "PrefixObjectSuffix".
        Falls back to ATLAS default naming on unknown releases.
        """
        try:
            if release_year.startswith("record_") and record_id is None:
                try:
                    record_id = int(release_year.split("_")[1])
                except (ValueError, IndexError):
                    pass

            schema = schemas.get_schema_for_release(release_year, record_id=record_id)
            naming_pattern = schema.get("naming_pattern", "dotted")
            object_mappings = schema.get("object_mappings", {})
            branch_obj_name = object_mappings.get(obj_name, obj_name)

            if naming_pattern == "flat":
                if field:
                    return f"{branch_obj_name}_{field}"
                return branch_obj_name
            else:
                prefix = schema["branch_prefix"]
                suffix = schema["branch_suffix"]
                return f"{prefix}{branch_obj_name}{suffix}"
        except KeyError:
            logging.warning(f"Release year '{release_year}' not found. Using default branch naming.")
            return "Analysis" + obj_name + "AuxDyn"

    @staticmethod
    def _find_cms_branches(
        base_branch_name: str,
        fields: list[str],
        obj_field_paths: dict,
        tree_branches: set[str]
    ) -> dict[str, str]:
        """
        Find CMS-style nested branches for a given object.

        CMS branch structure: {base}/{base}obj/{base}obj.{field_path}
        """
        branch_mappings = {}
        available_fields = []

        base_obj = f"{base_branch_name}obj"
        obj_container_patterns = [
            base_obj,
            f"{base_branch_name}/{base_obj}",
        ]
        has_obj_container = any(
            pattern in branch
            for branch in tree_branches
            for pattern in obj_container_patterns
        )
        if not has_obj_container:
            return {}

        for field in fields:
            if field not in obj_field_paths:
                continue

            field_path = obj_field_paths[field]
            field_indicator = consts.CMS_FIELD_INDICATORS.get(field)
            if not field_indicator:
                continue

            field_path_suffix = field_path[4:] if field_path.startswith("obj.") else field_path
            expected_path = f"{base_branch_name}/{base_obj}/{base_obj}.{field_path_suffix}"

            if expected_path in tree_branches:
                branch_mappings[expected_path] = field
                available_fields.append(field)
                continue

            matching = [
                branch for branch in tree_branches
                if branch.startswith(base_branch_name)
                and f"{base_obj}/" in branch
                and field_indicator in branch
            ]
            if matching:
                full_path = max(matching, key=len)
                branch_mappings[full_path] = field
                available_fields.append(field)
                continue

            field_selection_path = f"{base_obj}.{field_path_suffix}"
            branch_mappings[field_selection_path] = field
            available_fields.append(field)

        if FileParser._can_calculate_inv_mass(available_fields):
            return branch_mappings
        return {}

    @staticmethod
    def _extract_flat_branches(
        obj_name: str,
        fields: list[str],
        tree_branches: set[str],
        release_year: str
    ) -> dict[str, str]:
        """Extract branches using flat naming pattern (object_field)."""
        branch_base = FileParser._prepare_obj_branch_name(obj_name, release_year=release_year)
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
        branch_name = FileParser._prepare_obj_branch_name(obj_name, release_year=release_year)
        logging.debug(f"ATLAS-style naming for {obj_name}, branch base: {branch_name}")
        
        field_paths = schema_config.get("field_paths", {})
        obj_field_paths = field_paths.get(obj_name, {})
        
        if obj_field_paths:
            return FileParser._find_cms_branches(
                branch_name, fields, obj_field_paths, tree_branches
            )
        
        branch_to_quantity = {}
        available_fields = []
        
        for field in fields:
            branch_full = f"{branch_name}.{field}"
            if branch_full in tree_branches:
                available_fields.append(field)
                branch_to_quantity[branch_full] = field
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
        return ref_system.issubset(set(available_fields))
    
    @staticmethod
    def _filter_accessible_branches(
        tree,
        obj_branches: dict[str, dict[str, str]]
    ) -> dict[str, dict[str, str]]:
        """
        Test branch accessibility and filter out inaccessible ones.
        
        Reads ONE entry with ALL candidate branches at once to minimize
        HTTP round-trips for remote ROOT files.
        """
        all_candidate_branches = []
        for branch_mapping in obj_branches.values():
            all_candidate_branches.extend(branch_mapping.keys())
        
        accessible_set = set()
        try:
            test_arr = tree.arrays(
                all_candidate_branches,
                entry_start=0, entry_stop=1,
                library="ak"
            )
            accessible_set = set(test_arr.fields)
        except Exception:
            for branch_path in all_candidate_branches:
                try:
                    test_arr = tree.arrays(
                        branch_path,
                        entry_start=0, entry_stop=1,
                        library="ak"
                    )
                    if branch_path in test_arr.fields:
                        accessible_set.add(branch_path)
                except Exception:
                    continue
        
        accessible_obj_branches = {}
        for obj_name, branch_mapping in obj_branches.items():
            accessible_branches = {
                bp: qty for bp, qty in branch_mapping.items()
                if bp in accessible_set
            }
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
        obj_events_by_quantities = {
            obj_name: [] for obj_name in obj_branches.keys()
        }
        
        is_file_big = n_entries > batch_size
        if is_file_big:
            entry_ranges = [
                (start, min(start + batch_size, n_entries))
                for start in range(0, n_entries, batch_size)
            ]
        else:
            entry_ranges = [(0, n_entries)]
        
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
            
            for obj_name, branch_mapping in obj_branches.items():
                available_branches = [
                    b for b in branch_mapping.keys() if b in batch_data.fields
                ]
                if available_branches:
                    subset = batch_data[available_branches]
                    if len(subset) > 0:
                        obj_events_by_quantities[obj_name].append(subset)
        
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
        Attempts to find branches matching common patterns.
        """
        obj_branches = {}

        object_patterns = {
            "Electrons": ["Electron", "electron", "el"],
            "Muons": ["Muon", "muon", "mu"],
            "Jets": ["Jet", "jet"],
            "Photons": ["Photon", "photon", "gamma"]
        }

        required_fields = ["pt", "eta", "phi"]

        for obj_name, patterns in object_patterns.items():
            for pattern in patterns:
                matching_branches = [b for b in tree_branches if pattern.lower() in b.lower()]

                if matching_branches:
                    base_branch = None
                    for branch in matching_branches:
                        parts = branch.split(".")
                        if len(parts) == 2:
                            potential_base = parts[0]
                            has_required = all(
                                f"{potential_base}.{field}" in tree_branches
                                for field in required_fields
                            )
                            if has_required:
                                base_branch = potential_base
                                break

                    if base_branch:
                        available_fields = []
                        for field in required_fields + ["mass"]:
                            if f"{base_branch}.{field}" in tree_branches:
                                available_fields.append(field)

                        if FileParser._can_calculate_inv_mass(available_fields):
                            obj_branches[obj_name] = {
                                f"{base_branch}.{field}": field
                                for field in available_fields
                            }
                        break

        return obj_branches
