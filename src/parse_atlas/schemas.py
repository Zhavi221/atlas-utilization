"""
Schema definitions for different ATLAS Open Data release years.

Each release year may have different branch naming conventions.
Schemas use a template-based approach: prefix + object_name + suffix
"""

# Base object definitions (fields needed for invariant mass calculation)
BASE_OBJECTS = {
    "Electrons": ["pt", "eta", "phi", "mass"],
    "Muons": ["pt", "eta", "phi", "mass"],
    "Jets": ["pt", "eta", "phi", "mass"],
    "Photons": ["pt", "eta", "phi"]  # Photons typically don't have mass
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


def get_schema_for_release(release_year: str) -> dict:
    """
    Get the schema configuration for a specific release year.
    
    Automatically normalizes release years with _mc suffix (e.g., "2024r-pp_mc" -> "2024r-pp").
    
    Args:
        release_year: The release year identifier (e.g., "2024r-pp" or "2024r-pp_mc")
        
    Returns:
        Dictionary with 'branch_prefix', 'branch_suffix', and 'objects' keys
        
    Raises:
        KeyError: If release_year (after normalization) is not found in RELEASE_SCHEMAS
    """
    normalized_year = normalize_release_year(release_year)
    if normalized_year not in RELEASE_SCHEMAS:
        raise KeyError(
            f"Release year '{release_year}' (normalized: '{normalized_year}') not found in schemas. "
            f"Available releases: {list(RELEASE_SCHEMAS.keys())}"
        )
    return RELEASE_SCHEMAS[normalized_year]


def build_branch_name(obj_name: str, release_year: str = "2024r-pp", field: str = None) -> str:
    """
    Build a branch name using the template for a given release year.
    
    Automatically normalizes release years with _mc suffix.
    
    Args:
        obj_name: Canonical object name (e.g., "Electrons", "Muons")
        release_year: Release year identifier (e.g., "2024r-pp" or "2024r-pp_mc")
        field: Optional field name (e.g., "pt", "eta") - required for flat naming
        
    Returns:
        Full branch name:
        - For dotted naming: "AnalysisElectronsAuxDyn" (field not included)
        - For flat naming: "lep_pt" (field required)
    """
    schema = get_schema_for_release(release_year)  # Already normalizes _mc suffix
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
