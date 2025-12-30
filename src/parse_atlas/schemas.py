"""
Schema definitions for different ATLAS Open Data release years.

Each release year may have different branch naming conventions.
Schemas use a template-based approach: prefix + object_name + suffix
"""
import uproot
import requests
import json
from . import consts

# Base object definitions (fields needed for invariant mass calculation)
BASE_OBJECTS = {
    "Electrons": ["pt", "eta", "phi", "mass"],
    "Muons": ["pt", "eta", "phi", "mass"],
    "Jets": ["pt", "eta", "phi", "mass"],
    "Photons": ["pt", "eta", "phi"]  # Photons typically don't have mass
}

# Mapping from specific record IDs to their release year/schema identifier
# This will be populated when schemas are extracted from record IDs
RECORD_ID_TO_RELEASE = {
    30512: "cms-opendata",
    30546: "cms-opendata"
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
    "cms-opendata": {
        "naming_pattern": "dotted",  # patElectrons_slimmedElectrons__PAT./patElectrons_slimmedElectrons__PAT.obj/...
        "branch_prefix": "pat",
        "branch_suffix": "__PAT.",
        "object_mappings": {
            "Electrons": "Electrons_slimmedElectrons",
            "Muons": "Muons_slimmedMuons",
            "Jets": "Jets_slimmedJets",
            "Photons": "Photons_slimmedPhotons"
        },
        "objects": BASE_OBJECTS.copy(),
        "field_paths": {
            # CMS uses nested paths: base.obj.m_state.p4Polar_.fCoordinates.fPt
            # Full path example: patJets_slimmedJets__PAT./patJets_slimmedJets__PAT.obj/patJets_slimmedJets__PAT.obj.m_state.p4Polar_.fCoordinates.fPt
            "Electrons": {
                "pt": "obj.m_state.p4Polar_.fCoordinates.fPt",
                "eta": "obj.m_state.p4Polar_.fCoordinates.fEta",
                "phi": "obj.m_state.p4Polar_.fCoordinates.fPhi",
                "mass": "obj.m_state.p4Polar_.fCoordinates.fM"
            },
            "Muons": {
                "pt": "obj.m_state.p4Polar_.fCoordinates.fPt",
                "eta": "obj.m_state.p4Polar_.fCoordinates.fEta",
                "phi": "obj.m_state.p4Polar_.fCoordinates.fPhi",
                "mass": "obj.m_state.p4Polar_.fCoordinates.fM"
            },
            "Jets": {
                "pt": "obj.m_state.p4Polar_.fCoordinates.fPt",
                "eta": "obj.m_state.p4Polar_.fCoordinates.fEta",
                "phi": "obj.m_state.p4Polar_.fCoordinates.fPhi",
                "mass": "obj.m_state.p4Polar_.fCoordinates.fM"
            },
            "Photons": {
                "pt": "obj.m_state.p4Polar_.fCoordinates.fPt",
                "eta": "obj.m_state.p4Polar_.fCoordinates.fEta",
                "phi": "obj.m_state.p4Polar_.fCoordinates.fPhi"
            }
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
                # Dotted naming: AnalysisElectronsAuxDyn.pt
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
                prefixes = set()
                suffixes = set()
                object_names = {}
                
                for base in base_branches.keys():
                    # Try to extract prefix and suffix
                    # Common patterns: AnalysisElectronsAuxDyn, MuonsAuxDyn
                    if "Electron" in base or "electron" in base.lower():
                        obj_name = "Electrons"
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
                        object_names[obj_name] = obj_branch
                    elif "Muon" in base or "muon" in base.lower():
                        obj_name = "Muons"
                        if base.startswith("Analysis"):
                            schema["branch_prefix"] = "Analysis"
                            remaining = base[len("Analysis"):]
                        else:
                            remaining = base
                        if remaining.endswith("AuxDyn"):
                            schema["branch_suffix"] = "AuxDyn"
                            obj_branch = remaining[:-6]
                        else:
                            obj_branch = remaining
                        object_names[obj_name] = obj_branch
                    elif "Jet" in base or "jet" in base.lower():
                        obj_name = "Jets"
                        if base.startswith("Analysis"):
                            schema["branch_prefix"] = "Analysis"
                            remaining = base[len("Analysis"):]
                        else:
                            remaining = base
                        if remaining.endswith("AuxDyn"):
                            schema["branch_suffix"] = "AuxDyn"
                            obj_branch = remaining[:-6]
                        else:
                            obj_branch = remaining
                        object_names[obj_name] = obj_branch
                    elif "Photon" in base or "photon" in base.lower():
                        obj_name = "Photons"
                        if base.startswith("Analysis"):
                            schema["branch_prefix"] = "Analysis"
                            remaining = base[len("Analysis"):]
                        else:
                            remaining = base
                        if remaining.endswith("AuxDyn"):
                            schema["branch_suffix"] = "AuxDyn"
                            obj_branch = remaining[:-6]
                        else:
                            obj_branch = remaining
                        object_names[obj_name] = obj_branch
                
                # Build object mappings and objects dict
                for obj_name, obj_branch in object_names.items():
                    schema["object_mappings"][obj_name] = obj_branch
                    # Get available fields for this object
                    base_branch = f"{schema['branch_prefix']}{obj_branch}{schema['branch_suffix']}"
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
    If record_id is provided and release_year is "specific_records", will look up the schema
    for that record ID.
    
    Args:
        release_year: The release year identifier (e.g., "2024r-pp" or "2024r-pp_mc" or "specific_records")
        record_id: Optional record ID to use when release_year is "specific_records"
        
    Returns:
        Dictionary with 'branch_prefix', 'branch_suffix', and 'objects' keys
        
    Raises:
        KeyError: If release_year (after normalization) is not found in RELEASE_SCHEMAS
    """
    # Handle specific record IDs
    if release_year == "specific_records" and record_id is not None:
        # Check if we have a mapping for this record ID
        if record_id in RECORD_ID_TO_RELEASE:
            release_year = RECORD_ID_TO_RELEASE[record_id]
        else:
            # Try to extract schema on the fly (this will be slow)
            import logging
            logging.warning(f"Record ID {record_id} not in mapping, extracting schema on the fly...")
            schema = extract_schema_from_record_id(record_id)
            # Create a unique identifier for this record ID's schema
            schema_key = f"record_{record_id}"
            RELEASE_SCHEMAS[schema_key] = schema
            RECORD_ID_TO_RELEASE[record_id] = schema_key
            return schema
    
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
        release_year: Release year identifier (e.g., "2024r-pp" or "2024r-pp_mc" or "specific_records")
        field: Optional field name (e.g., "pt", "eta") - required for flat naming
        record_id: Optional record ID to use when release_year is "specific_records"
        
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
