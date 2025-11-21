import uproot

def find_resolution_info(filename, treename):
    """Find resolution/uncertainty branches"""
    file = uproot.open(filename)
    tree = file[treename]
    
    # Keywords specific to resolution/uncertainty
    resolution_keywords = [
        'resolution', 'Resolution', 'reso',
        'sigma', 'Sigma',
        'error', 'Error',
        'uncertainty', 'Uncertainty',
        'smear', 'Smear',
        'JER',  # Jet Energy Resolution
        'cov',  # Covariance
        'width'
    ]
    
    matching = []
    for branch_name in tree.keys():
        if any(kw in branch_name for kw in resolution_keywords):
            matching.append(branch_name)
    
    print("Resolution/Uncertainty branches:")
    for branch in sorted(matching):
        print(f"  {branch}")
    
    return matching

def find_object_uncertainties(filename, treename, object_type="Jet"):
    """Find uncertainty branches for specific physics objects"""
    file = uproot.open(filename)
    tree = file[treename]
    
    # Get all branches for this object type
    obj_branches = [k for k in tree.keys() if object_type in k and 'Aux.' in k]
    
    # Look for ones that might be uncertainties
    uncertainty_patterns = ['err', 'Err', 'unc', 'Unc', 'sigma', 'cov', 'resolution']
    
    print(f"\n{object_type} uncertainty-related branches:")
    for branch in sorted(obj_branches):
        if any(pat in branch for pat in uncertainty_patterns):
            print(f"  {branch}")
            try:
                value = tree[branch].array(entry_stop=1)[0]
                print(f"    First value: {value}\n")
            except:
                print(f"    (cannot read)\n")

def show_all_aux_branches(filename, treename, object_prefix="Electrons"):
    """Show all auxiliary branches for an object - resolution data lives here"""
    file = uproot.open(filename)
    tree = file[treename]
    
    aux_branches = [k for k in tree.keys() 
                    if k.startswith(object_prefix) and 'Aux.' in k]
    
    print(f"\nAll {object_prefix} Aux branches:")
    print("=" * 70)
    
    for branch in sorted(aux_branches):
        try:
            arr = tree[branch].array(entry_stop=5)  # First 5 entries
            print(f"\n{branch}:")
            print(f"  Type: {tree[branch].typename}")
            print(f"  First entry: {arr[0]}")
        except:
            print(f"\n{branch}: (cannot read)")

import uproot

def find_calibration_branches(filename, treename):
    """Find branches related to calibration and resolution"""
    file = uproot.open(filename)
    tree = file[treename]
    
    # Keywords to search for
    keywords = [
        'calib', 'Calib', 'CALIB',
        'resolution', 'Resolution', 'reso',
        'scale', 'Scale',
        'smear', 'Smear',
        'correction', 'Correction',
        'systematic', 'Systematic',
        'JES', 'JER',  # Jet Energy Scale/Resolution
        'EES', 'EER',  # Electron Energy Scale/Resolution
        'MES', 'MER',  # Muon Energy Scale/Resolution
        'EM',          # Electromagnetic scale
        'LC',          # Local cell weighting
    ]
    
    matching_branches = []
    
    for branch_name in tree.keys():
        if any(keyword in branch_name for keyword in keywords):
            matching_branches.append(branch_name)
    
    print("=" * 70)
    print("Calibration/Resolution related branches:")
    print("=" * 70)
    
    for branch in sorted(matching_branches):
        try:
            branch_obj = tree[branch]
            print(f"{branch:50s} | {branch_obj.typename}")
        except:
            print(f"{branch:50s} | (unable to read type)")
    
    return matching_branches

# Use it