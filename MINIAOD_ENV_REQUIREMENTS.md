# MiniAOD Environment Requirements Assessment

## Current Environment Status

### ✅ Available in Your Environment

1. **PyROOT (ROOT with Python bindings)**
   - Version: ROOT 6.32.06
   - Location: `/cvmfs/sft.cern.ch/lcg/releases/ROOT/6.32.06-9932a/x86_64-el9-gcc14-opt`
   - Status: **FULLY FUNCTIONAL**
   - Can open ROOT files via xrootd ✓
   - Can access TTree and branches ✓

2. **LCG View Setup**
   - View: `LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt`
   - Provides: ROOT, Python, and other scientific libraries
   - Status: **ACTIVE**

3. **uproot Library**
   - Version: 5.6.2
   - Status: **AVAILABLE** (works for Jets, not for pat::Electron C++ objects)

### ❌ Not Available in Your Environment

1. **CMSSW Framework**
   - CMSSW_BASE: NOT SET
   - CMSSW_VERSION: NOT SET
   - Status: **NOT AVAILABLE**

2. **CMS EDM Dictionaries**
   - pat::Electron: NOT available
   - pat::Muon: NOT available
   - pat::Photon: NOT available
   - pat::Jet: NOT available (but Jets work via uproot due to exposed fields)
   - Status: **NOT LOADED**

## Test Results

### File Access Test
- ✅ Successfully opened MiniAOD file via xrootd
- ✅ Can access "Events" tree
- ✅ Found 23,367 entries
- ✅ Found 107 branches total
- ✅ Found 3 Electron-related branches:
  - `patElectrons_slimmedElectrons__PAT.`
  - `patElectrons_slimmedLowPtElectrons__PAT.`
  - `recoGsfElectronCores_reducedEgamma_reducedGedGsfElectronCores_PAT.`

### Electron Branch Access Test
- ✅ Can get branch object
- ✅ Branch type identified: `edm::Wrapper<vector<pat::Electron> >`
- ❌ Cannot directly access collection without CMSSW dictionaries
- ❌ Cannot use pat::Electron C++ methods

## What This Means

### Current Capabilities
1. **Jets**: Can be read via `uproot` because Jets have their nested fields (pt, eta, phi, m) exposed in the ROOT tree structure
2. **Electrons/Muons/Photons**: Cannot be read because:
   - They are stored as C++ objects (`pat::Electron`, etc.)
   - `uproot` cannot interpret these C++ objects
   - PyROOT can see the branches but cannot access the data without CMSSW dictionaries

### Limitations
- Without CMSSW dictionaries, PyROOT cannot:
  - Access `pat::Electron` collection elements
  - Call C++ methods like `.pt()`, `.eta()`, `.phi()`, `.mass()`
  - Iterate over the collection

## Options for MiniAOD Access

### Option 1: Use CMSSW Environment (Recommended for Full Access)
**Requirements:**
- Install CMSSW (10-30 GB disk space)
- For 2016 data: CMSSW_10_6_30
- Setup: `cmsrel CMSSW_10_6_30 && cd CMSSW_10_6_30/src && cmsenv`

**Pros:**
- Full access to all MiniAOD objects
- Can use pat::Electron, pat::Muon, pat::Photon C++ methods
- Official CMS analysis environment

**Cons:**
- Large installation size
- Requires compilation time
- More complex setup

### Option 2: Use CMS Docker Container
**Requirements:**
- Docker installed
- Pull CMS Open Data container

**Pros:**
- Pre-configured environment
- No manual installation
- Isolated from host system

**Cons:**
- Requires Docker
- May have performance overhead

### Option 3: Use CERN Virtual Machine
**Requirements:**
- VirtualBox or compatible VM software
- Download CMS VM (~20-30 GB)

**Pros:**
- Complete CMS environment
- Pre-configured
- Good for development

**Cons:**
- Large download
- VM overhead

### Option 4: Hybrid Approach (Current + PyROOT Workaround)
**Requirements:**
- Keep current LCG view setup
- Use PyROOT for low-level branch access
- May need to parse binary data manually

**Pros:**
- No additional installation
- Uses existing environment

**Cons:**
- Limited functionality
- Complex implementation
- May not work for all fields

## Recommendation

**For your current setup**, you have two viable paths:

1. **Short-term**: Continue using `uproot` for Jets (which works) and accept that Electrons/Muons/Photons are not accessible in MiniAOD without CMSSW.

2. **Long-term**: Set up a CMSSW environment (via Docker, VM, or direct installation) to get full MiniAOD access. This would allow you to:
   - Access all particle types
   - Use C++ methods for all objects
   - Have full compatibility with CMS analysis tools

## Next Steps

If you want to proceed with CMSSW setup, I can help you:
1. Set up a CMSSW Docker container
2. Configure your submission scripts to use CMSSW environment
3. Modify your parser to use PyROOT with CMSSW for Electrons/Muons/Photons
4. Keep uproot for Jets (hybrid approach)

## Environment Setup Commands (Current)

Your current environment setup in submission scripts:
```bash
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"
```

This gives you:
- ✅ PyROOT
- ✅ ROOT 6.32.06
- ✅ Python 3.11
- ✅ uproot 5.6.2
- ❌ CMSSW dictionaries

