"""
Schema definitions for different ATLAS Open Data release years.

Each release year may have different branch naming conventions.
Schemas use a template-based approach: prefix + object_name + suffix
"""
import uproot
import requests
import json
from services import consts

# Base object definitions (fields needed for invariant mass calculation)
BASE_OBJECTS = {
    "Electrons": ["pt", "eta", "phi", "mass"],
    "Muons": ["pt", "eta", "phi", "mass"],
    "Jets": ["pt", "eta", "phi", "mass"],
    "Photons": ["pt", "eta", "phi"]  # Photons typically don't have mass
}

# Mapping from specific record IDs to their release year/schema identifier
# This will be populated when schemas are extracted from record IDs
RECORD_ID_TO_SCHEMA = {
    30529: "cms-nanoaod",  # NanoAOD format
    30562: "cms-nanoaod",  # NanoAOD format
    30530: "cms-nanoaod",  # NanoAOD format
    30563: "cms-nanoaod",  # NanoAOD format
}

# Release-specific branch naming templates
# Each schema can use:
# - "naming_pattern": "dotted" (object.field) or "flat" (object_field)
# - "branch_prefix": prefix before object name (empty string if none)
# - "branch_suffix": suffix after object name (empty string if none)
# - "object_mappings": map from canonical object names to branch object names
# - "objects": dict of canonical object names to required fields
RELEASE_SCHEMAS = {
    "2024r-pp": {
        "naming_pattern": "dotted",  # AnalysisElectronsAuxDyn.pt
        "branch_prefix": "Analysis",
        "branch_suffix": "AuxDyn",
        "object_mappings": {
            "Electrons": "Electrons",
            "Muons": "Muons",
            "Jets": "Jets",
            "Photons": "Photons"
        },
        "objects": BASE_OBJECTS.copy()
    },
    "cms-nanoaod": {
        "naming_pattern": "flat",  # Electron_pt, Electron_eta, Muon_pt, etc.
        "branch_prefix": "",
        "branch_suffix": "",
        "object_mappings": {
            "Electrons": "Electron",  # Maps to Electron_pt, Electron_eta, etc.
            "Muons": "Muon",          # Maps to Muon_pt, Muon_eta, etc.
            "Jets": "Jet",            # Maps to Jet_pt, Jet_eta, etc.
            "Photons": "Photon"       # Maps to Photon_pt, Photon_eta, etc.
        },
        "objects": {
            "Electrons": ["pt", "eta", "phi", "mass"],
            "Muons": ["pt", "eta", "phi", "mass"],
            "Jets": ["pt", "eta", "phi", "mass"],
            "Photons": ["pt", "eta", "phi", "mass"]  # NanoAOD includes Photon_mass
        }
    },
    "2024r-hi": {
        "naming_pattern": "dotted",  # MuonsAuxDyn.pt
        "branch_prefix": "",  # No prefix for Muons
        "branch_suffix": "AuxDyn",
        "object_mappings": {
            "Muons": "Muons",  # Only Muons detected in this release
            # Note: Electrons, Jets, Photons may not be available
        },
        "objects": {
            "Muons": ["pt", "eta", "phi", "mass"]
        }
    },
    "2020e-13tev": {
        "naming_pattern": "dotted",  # Assumed similar to 2024r-pp
        "branch_prefix": "Analysis",
        "branch_suffix": "AuxDyn",
        "object_mappings": {
            "Electrons": "Electrons",
            "Muons": "Muons",
            "Jets": "Jets",
            "Photons": "Photons"
        },
        "objects": BASE_OBJECTS.copy()
        # Note: Branch names may differ - needs verification via inspection
    },
    "2016e-8tev": {
        "naming_pattern": "flat",  # lep_pt, jet_pt
        "branch_prefix": "",
        "branch_suffix": "",
        "object_mappings": {
            # Note: "lep" represents combined leptons (electrons + muons)
            # Both Electrons and Muons will extract from the same "lep_*" branches
            # This is intentional - the data contains combined lepton information
            "Electrons": "lep",
            "Muons": "lep",
            "Jets": "jet"
            # Note: Photons not detected in this release
        },
        "objects": {
            "Electrons": ["pt", "eta", "phi"],  # No mass field detected
            "Muons": ["pt", "eta", "phi"],      # No mass field detected
            "Jets": ["pt", "eta", "phi", "mass"]
        }
    },
    "2025e-13tev-beta": {
        "naming_pattern": "flat",  # lep_pt, jet_pt, photon_pt, tau_pt
        "branch_prefix": "",
        "branch_suffix": "",
        "object_mappings": {
            # Note: "lep" represents combined leptons (electrons + muons)
            # Both Electrons and Muons will extract from the same "lep_*" branches
            "Electrons": "lep",
            "Muons": "lep",
            "Jets": "jet",
            "Photons": "photon"
            # Note: tau_pt exists but represents tau jets, not regular jets
        },
        "objects": {
            "Electrons": ["pt", "eta", "phi"],  # No mass field detected
            "Muons": ["pt", "eta", "phi"],      # No mass field detected
            "Jets": ["pt", "eta", "phi"],       # No mass field detected
            "Photons": ["pt", "eta", "phi"]
        }
    },
    "2025r-evgen-13tev": {
        "naming_pattern": "dotted",  # Assumed similar to 2024r-pp
        "branch_prefix": "Analysis",
        "branch_suffix": "AuxDyn",
        "object_mappings": {
            "Electrons": "Electrons",
            "Muons": "Muons",
            "Jets": "Jets",
            "Photons": "Photons"
        },
        "objects": BASE_OBJECTS.copy()
        # Note: Files are gzipped, schema inferred from similar releases
    },
    "2025r-evgen-13p6tev": {
        "naming_pattern": "dotted",  # Assumed similar to 2024r-pp
        "branch_prefix": "Analysis",
        "branch_suffix": "AuxDyn",
        "object_mappings": {
            "Electrons": "Electrons",
            "Muons": "Muons",
            "Jets": "Jets",
            "Photons": "Photons"
        },
        "objects": BASE_OBJECTS.copy()
        # Note: Files are gzipped, schema inferred from similar releases
    }
}

# Legacy schema for backward compatibility
INVARIANT_MASS_SCHEMA = BASE_OBJECTS.copy()

# Legacy random schema (kept for compatibility)
SCHEMA_RANDOM = {
    "electron": ["pt", "eta", "phi", "mass"],
    "muon": ["pt", "eta", "phi", "mass"],
    "jet": ["pt", "eta", "phi", "mass"],
    "photon": ["pt", "eta", "phi"]
}


def normalize_release_year(release_year: str) -> str:
    """
    Normalize release year by removing _mc suffix.
    
    The _mc suffix is used for file organization/view separation only.
    For schema lookups and parsing logic, we treat "_mc" release years
    the same as their base release year.
    
    Args:
        release_year: Release year identifier (e.g., "2024r-pp" or "2024r-pp_mc")
        
    Returns:
        Normalized release year without _mc suffix (e.g., "2024r-pp")
    """
    if release_year.endswith("_mc"):
        return release_year[:-3]  # Remove "_mc" suffix
    return release_year


def extract_schema_from_record_id(record_id: int, sample_file_uri: str = None) -> dict:
    """
    Extract schema from a specific record ID by inspecting a sample file.
    
    This function downloads and inspects a ROOT file from the record to determine:
    - Naming pattern (dotted vs flat)
    - Branch prefix and suffix
    - Available objects and their fields
    - Object mappings
    
    Args:
        record_id: The CERN Open Data record ID
        sample_file_uri: Optional URI to a specific file. If None, uses first file from record.
        
    Returns:
        Dictionary with schema configuration matching RELEASE_SCHEMAS format
    """
    import logging
    
    # Get file URI if not provided
    if sample_file_uri is None:
        r = requests.get(consts.CMS_RECID_FILEPAGE_URL.format(record_id))
        json_r = json.loads(r.text)
        if not json_r.get("index_files", {}).get("files"):
            raise ValueError(f"No files found in record {record_id}")
        # Get first file URI
        sample_file_uri = json_r["index_files"]["files"][0]["files"][0]["uri"]
    
    logging.info(f"Extracting schema from record {record_id} using file: {sample_file_uri}")
    
    # Open the ROOT file and inspect branches
    try:
        with uproot.open(sample_file_uri) as file:
            # Try common tree names
            tree_names = ["CollectionTree", "mini", "analysis"]
            tree = None
            for tree_name in tree_names:
                if tree_name in file:
                    tree = file[tree_name]
                    break
            
            if tree is None:
                # Use first available tree
                tree = file[list(file.keys())[0]]
            
            all_branches = set(tree.keys())
            logging.info(f"Found {len(all_branches)} branches in tree")
            
            # Determine naming pattern
            naming_pattern = "dotted"  # default
            has_dotted = any("." in branch for branch in all_branches)
            has_flat = any("_" in branch and "." not in branch for branch in all_branches)
            
            if has_flat and not has_dotted:
                naming_pattern = "flat"
            elif has_dotted:
                naming_pattern = "dotted"
            
            # Extract schema based on naming pattern
            schema = {
                "naming_pattern": naming_pattern,
                "branch_prefix": "",
                "branch_suffix": "",
                "object_mappings": {},
                "objects": {}
            }
            
            if naming_pattern == "dotted":
                # Dotted naming: AnalysisElectronsAuxDyn.pt or patElectrons_slimmedElectrons__PAT./...
                # Find common prefixes and suffixes
                dotted_branches = [b for b in all_branches if "." in b]
                
                # Group by base branch name
                base_branches = {}
                for branch in dotted_branches:
                    base, field = branch.rsplit(".", 1)
                    if base not in base_branches:
                        base_branches[base] = []
                    base_branches[base].append(field)
                
                # Detect prefix and suffix patterns
                object_names = {}
                cms_pattern_detected = False
                
                # First, try to detect CMS pattern: pat{Object}_slimmed{Object}__PAT.
                for base in base_branches.keys():
                    # Check for CMS pattern: pat{Object}_slimmed{Object}__PAT.
                    if base.startswith("pat") and "__PAT." in base:
                        cms_pattern_detected = True
                        schema["branch_prefix"] = "pat"
                        schema["branch_suffix"] = "__PAT."
                        
                        # Extract object name from pattern like: patJets_slimmedJets__PAT.
                        # The base might be: patJets_slimmedJets__PAT. or patJets_slimmedJets__PAT./patJets_slimmedJets__PAT.obj/...
                        # Find the first occurrence of __PAT. to extract the object name
                        pat_index = base.find("__PAT.")
                        if pat_index > 0:
                            # Extract: patJets_slimmedJets (everything before __PAT.)
                            remaining = base[len("pat"):pat_index]
                        else:
                            # Fallback: remove pat prefix
                            remaining = base[len("pat"):]
                            if remaining.endswith("__PAT."):
                                remaining = remaining[:-6]  # Remove "__PAT."
                        
                        # Pattern matching for all particle types using the same template
                        particle_patterns = [
                            ("Electron", "Electrons", "Electrons_slimmedElectrons"),
                            ("Muon", "Muons", "Muons_slimmedMuons"),
                            ("Photon", "Photons", "Photons_slimmedPhotons"),
                            ("Jet", "Jets", "Jets_slimmedJets")
                        ]
                        
                        for pattern, obj_name, default_mapping in particle_patterns:
                            if pattern in remaining or pattern.lower() in remaining.lower():
                                # Handle variants: prefer standard over variants (e.g., slimmedJets over slimmedJetsPuppi)
                                if obj_name not in object_names:
                                    object_names[obj_name] = remaining
                                elif "Puppi" not in remaining or "Puppi" in object_names.get(obj_name, ""):
                                    # Prefer non-Puppi variant, or keep current if both are Puppi
                                    if "Puppi" not in remaining:
                                        object_names[obj_name] = remaining
                                break
                
                # If CMS pattern not detected, try ATLAS pattern
                if not cms_pattern_detected:
                    for base in base_branches.keys():
                        # ATLAS patterns: AnalysisElectronsAuxDyn, MuonsAuxDyn
                        particle_patterns = [
                            ("Electron", "Electrons"),
                            ("Muon", "Muons"),
                            ("Jet", "Jets"),
                            ("Photon", "Photons")
                        ]
                        
                        for pattern, obj_name in particle_patterns:
                            if pattern in base or pattern.lower() in base.lower():
                                # Extract prefix (e.g., "Analysis")
                                if base.startswith("Analysis"):
                                    schema["branch_prefix"] = "Analysis"
                                    remaining = base[len("Analysis"):]
                                else:
                                    remaining = base
                                
                                # Extract suffix (e.g., "AuxDyn")
                                if remaining.endswith("AuxDyn"):
                                    schema["branch_suffix"] = "AuxDyn"
                                    obj_branch = remaining[:-6]  # Remove "AuxDyn"
                                else:
                                    obj_branch = remaining
                                
                                if obj_name not in object_names:
                                    object_names[obj_name] = obj_branch
                                break
                
                # Build object mappings and objects dict
                # Initialize field_paths for CMS patterns
                if cms_pattern_detected:
                    schema["field_paths"] = {}
                
                for obj_name, obj_branch in object_names.items():
                    schema["object_mappings"][obj_name] = obj_branch
                    # Get available fields for this object
                    base_branch = f"{schema['branch_prefix']}{obj_branch}{schema['branch_suffix']}"
                    
                    if cms_pattern_detected:
                        # For CMS, fields are nested: patJets_slimmedJets__PAT.obj.m_state.p4Polar_.fCoordinates.fPt
                        # Check if base branch exists and look for nested field patterns
                        relevant_fields = []
                        field_paths = {}
                        has_matching_branches = False
                        
                        # Check for standard CMS field patterns in branches
                        for branch in dotted_branches:
                            if branch.startswith(base_branch):
                                has_matching_branches = True
                                # Check for standard CMS field patterns
                                if "fPt" in branch or "fCoordinates.fPt" in branch:
                                    if "pt" not in relevant_fields:
                                        relevant_fields.append("pt")
                                        field_paths["pt"] = "obj.m_state.p4Polar_.fCoordinates.fPt"
                                if "fEta" in branch or "fCoordinates.fEta" in branch:
                                    if "eta" not in relevant_fields:
                                        relevant_fields.append("eta")
                                        field_paths["eta"] = "obj.m_state.p4Polar_.fCoordinates.fEta"
                                if "fPhi" in branch or "fCoordinates.fPhi" in branch:
                                    if "phi" not in relevant_fields:
                                        relevant_fields.append("phi")
                                        field_paths["phi"] = "obj.m_state.p4Polar_.fCoordinates.fPhi"
                                if "fM" in branch or "fCoordinates.fM" in branch:
                                    if "mass" not in relevant_fields:
                                        relevant_fields.append("mass")
                                        field_paths["mass"] = "obj.m_state.p4Polar_.fCoordinates.fM"
                        
                        # If no fields detected but we have matching branches, use default CMS field paths
                        if not relevant_fields and has_matching_branches:
                            # Default CMS field paths for all standard fields
                            relevant_fields = ["pt", "eta", "phi", "mass"]
                            field_paths = {
                                "pt": "obj.m_state.p4Polar_.fCoordinates.fPt",
                                "eta": "obj.m_state.p4Polar_.fCoordinates.fEta",
                                "phi": "obj.m_state.p4Polar_.fCoordinates.fPhi",
                                "mass": "obj.m_state.p4Polar_.fCoordinates.fM"
                            }
                        
                        if relevant_fields:
                            schema["objects"][obj_name] = relevant_fields
                            schema["field_paths"][obj_name] = field_paths
                    else:
                        # For ATLAS, fields are direct: AnalysisElectronsAuxDyn.pt
                        available_fields = base_branches.get(base_branch, [])
                        # Filter to only include relevant fields
                        relevant_fields = [f for f in available_fields if f in ["pt", "eta", "phi", "mass"]]
                        if relevant_fields:
                            schema["objects"][obj_name] = relevant_fields
                
            else:  # flat naming
                # Flat naming: lep_pt, jet_pt
                flat_branches = [b for b in all_branches if "_" in b and "." not in b]
                
                # Group by object prefix
                object_prefixes = {}
                for branch in flat_branches:
                    parts = branch.split("_")
                    if len(parts) >= 2:
                        prefix = parts[0]
                        field = parts[1]
                        if prefix not in object_prefixes:
                            object_prefixes[prefix] = []
                        object_prefixes[prefix].append(field)
                
                # Map common prefixes to object names
                prefix_to_obj = {
                    "lep": ["Electrons", "Muons"],
                    "electron": ["Electrons"],
                    "muon": ["Muons"],
                    "jet": ["Jets"],
                    "photon": ["Photons"]
                }
                
                for prefix, fields in object_prefixes.items():
                    # Find matching object names
                    for pattern, obj_names in prefix_to_obj.items():
                        if pattern in prefix.lower():
                            for obj_name in obj_names:
                                if obj_name not in schema["object_mappings"]:
                                    schema["object_mappings"][obj_name] = prefix
                                # Get relevant fields
                                relevant_fields = [f for f in fields if f in ["pt", "eta", "phi", "mass"]]
                                if relevant_fields and obj_name not in schema["objects"]:
                                    schema["objects"][obj_name] = relevant_fields
                            break
            
            logging.info(f"Extracted schema for record {record_id}: {schema}")
            return schema
            
    except Exception as e:
        import logging
        logging.error(f"Error extracting schema from record {record_id}: {e}")
        raise


def get_schema_for_release(release_year: str, record_id: int = None) -> dict:
    """
    Get the schema configuration for a specific release year or record ID.
    
    Automatically normalizes release years with _mc suffix (e.g., "2024r-pp_mc" -> "2024r-pp").
    If release_year is in "record_{record_id}" format, will look up the schema for that record ID
    from the RECORD_ID_TO_SCHEMA mapping.
    
    Args:
        release_year: The release year identifier (e.g., "2024r-pp", "2024r-pp_mc", or "record_30512")
        record_id: Optional record ID. If None and release_year starts with "record_", will extract from release_year
        
    Returns:
        Dictionary with 'branch_prefix', 'branch_suffix', and 'objects' keys
        
    Raises:
        KeyError: If release_year (after normalization) is not found in RELEASE_SCHEMAS,
                 or if record_id is not found in RECORD_ID_TO_SCHEMA mapping
    """
    # Handle record IDs (release_year format: "record_{record_id}")
    if release_year.startswith("record_"):
        # Extract record_id from release_year if not provided
        if record_id is None:
            try:
                record_id = int(release_year.split("_")[1])
            except (ValueError, IndexError):
                pass
        
        if record_id is not None:
            # Check if we have a mapping for this record ID
            if record_id in RECORD_ID_TO_SCHEMA:
                release_year = RECORD_ID_TO_SCHEMA[record_id]
            else:
                raise KeyError(
                    f"Record ID {record_id} not found in RECORD_ID_TO_SCHEMA mapping. "
                    f"Available record IDs: {list(RECORD_ID_TO_SCHEMA.keys())}"
                )
    
    normalized_year = normalize_release_year(release_year)
    if normalized_year not in RELEASE_SCHEMAS:
        raise KeyError(
            f"Release year '{release_year}' (normalized: '{normalized_year}') not found in schemas. "
            f"Available releases: {list(RELEASE_SCHEMAS.keys())}"
        )
    return RELEASE_SCHEMAS[normalized_year]


def build_branch_name(obj_name: str, release_year: str = "2024r-pp", field: str = None, record_id: int = None) -> str:
    """
    Build a branch name using the template for a given release year.
    
    Automatically normalizes release years with _mc suffix.
    
    Args:
        obj_name: Canonical object name (e.g., "Electrons", "Muons")
        release_year: Release year identifier (e.g., "2024r-pp", "2024r-pp_mc", or "record_30512")
        field: Optional field name (e.g., "pt", "eta") - required for flat naming
        record_id: Optional record ID. If None and release_year starts with "record_", will extract from release_year
        
    Returns:
        Full branch name:
        - For dotted naming: "AnalysisElectronsAuxDyn" (field not included)
        - For flat naming: "lep_pt" (field required)
    """
    schema = get_schema_for_release(release_year, record_id=record_id)  # Already normalizes _mc suffix
    naming_pattern = schema.get("naming_pattern", "dotted")
    object_mappings = schema.get("object_mappings", {})
    
    # Get the branch object name (may differ from canonical name)
    branch_obj_name = object_mappings.get(obj_name, obj_name)
    
    if naming_pattern == "flat":
        if field is None:
            raise ValueError(f"Field name required for flat naming pattern in release {release_year}")
        return f"{branch_obj_name}_{field}"
    else:  # dotted naming
        prefix = schema["branch_prefix"]
        suffix = schema["branch_suffix"]
        return f"{prefix}{branch_obj_name}{suffix}"


def get_available_releases() -> list:
    """Return list of all available release years."""
    return list(RELEASE_SCHEMAS.keys())
