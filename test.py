#!/usr/bin/env python3
"""
XRootD SSL Certificate Fixes for ATLAS Data Access
Solutions to bypass certificate verification issues
"""

import os
import sys

def print_banner(text):
    print("\n" + "="*60)
    print(f"🔧 {text}")
    print("="*60)

def setup_xrootd_ssl_bypass():
    """Set up XRootD environment to bypass SSL certificate checks"""
    print_banner("XRootD SSL Bypass Setup")
    
    # XRootD environment variables to bypass certificate verification
    xrd_env_vars = {
        'XRD_TLSNOCHECK': '1',              # Skip TLS certificate verification
        'XRD_TLSNOVERIFY': '1',             # Skip TLS hostname verification  
        'XRD_TLSMETALINK': '0',             # Disable metalink for TLS
        'XRD_CONNECTIONRETRY': '5',         # Retry connections
        'XRD_REQUESTTIMEOUT': '300',        # 5 minute timeout
        'XRD_TIMEOUTRESOLUTION': '1',       # Timeout resolution
        'XRD_NETWORKSTACK': 'IPAuto',       # Network stack
    }
    
    print("Setting XRootD environment variables:")
    for var, value in xrd_env_vars.items():
        os.environ[var] = value
        print(f"  export {var}={value}")
    
    return xrd_env_vars

def test_uproot_with_ssl_bypass():
    """Test uproot with SSL bypass"""
    print_banner("Testing uproot with SSL bypass")
    
    try:
        import uproot
        print(f"✅ uproot {uproot.__version__} available")
        
        # ATLAS file URL
        atlas_url = "root://eospublic.cern.ch//eos/opendata/atlas/rucio/data16_13TeV/DAOD_PHYSLITE.37019975._000235.pool.root.1"
        
        print(f"🔄 Testing XRootD access with SSL bypass...")
        print(f"URL: {atlas_url}")
        
        # Try with different configurations
        configs = [
            {"name": "Basic SSL bypass", "timeout": 120, "num_fails": 5},
            {"name": "Explicit XRootD source", "timeout": 180, "num_fails": 8, "file_handler": uproot.source.xrootd.XRootDSource},
            {"name": "Long timeout", "timeout": 300, "num_fails": 10}
        ]
        
        for config in configs:
            print(f"\n🧪 {config['name']}:")
            try:
                with uproot.open(atlas_url, **{k: v for k, v in config.items() if k != 'name'}) as file:
                    keys = list(file.keys())
                    print(f"   ✅ SUCCESS! Found {len(keys)} objects")
                    
                    if "CollectionTree" in file:
                        tree = file["CollectionTree"]
                        print(f"   🌳 CollectionTree: {len(tree.keys())} branches, {tree.num_entries} entries")
                    
                    return True
                    
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue
        
        return False
        
    except ImportError:
        print("❌ uproot not available")
        return False

def test_root_with_ssl_bypass():
    """Test ROOT with SSL bypass"""
    print_banner("Testing ROOT with SSL bypass")
    
    try:
        import ROOT
        print(f"✅ ROOT {ROOT.gROOT.GetVersion()} available")
        
        # Set ROOT to ignore SSL errors
        ROOT.gEnv.SetValue("XNet.ConnectTimeout", 300)
        ROOT.gEnv.SetValue("XNet.RequestTimeout", 300)
        ROOT.gEnv.SetValue("XNet.TransactionTimeout", 300)
        
        atlas_url = "root://eospublic.cern.ch//eos/opendata/atlas/rucio/data16_13TeV/DAOD_PHYSLITE.37019975._000235.pool.root.1"
        
        print(f"🔄 Testing ROOT TFile with SSL bypass...")
        
        # Suppress ROOT error messages temporarily
        ROOT.gErrorIgnoreLevel = ROOT.kWarning
        
        tfile = ROOT.TFile.Open(atlas_url)
        
        if tfile and not tfile.IsZombie():
            keys = tfile.GetListOfKeys()
            print(f"✅ ROOT SUCCESS! Found {keys.GetSize()} objects")
            
            tree = tfile.Get("CollectionTree")
            if tree:
                print(f"🌳 CollectionTree: {tree.GetEntries()} entries")
            
            tfile.Close()
            return True
        else:
            print("❌ ROOT failed: file is zombie or couldn't open")
            return False
            
    except ImportError:
        print("❌ ROOT not available")
        return False
    except Exception as e:
        print(f"❌ ROOT failed: {e}")
        return False

def create_working_script(method):
    """Create working script based on successful method"""
    
    script_content = f'''#!/usr/bin/env python3
"""
Working ATLAS XRootD Access Script
Bypasses SSL certificate verification issues
Method: {method}
"""

import os

def setup_xrootd_environment():
    """Set up XRootD environment to bypass SSL issues"""
    xrd_env = {{
        'XRD_TLSNOCHECK': '1',              # Skip TLS certificate verification
        'XRD_TLSNOVERIFY': '1',             # Skip TLS hostname verification
        'XRD_CONNECTIONRETRY': '5',         # Retry connections
        'XRD_REQUESTTIMEOUT': '300',        # 5 minute timeout
        'XRD_NETWORKSTACK': 'IPAuto',       # Network stack
    }}
    
    for var, value in xrd_env.items():
        os.environ[var] = value
    
    print("🔧 XRootD SSL bypass configured")

def atlas_analysis():
    """ATLAS data analysis with SSL bypass"""
    
    # Configure environment first
    setup_xrootd_environment()
    
    atlas_url = "root://eospublic.cern.ch//eos/opendata/atlas/rucio/data16_13TeV/DAOD_PHYSLITE.37019975._000235.pool.root.1"
    
'''
    
    if method == "uproot":
        script_content += '''
    # Method: uproot with SSL bypass
    try:
        import uproot
        
        print("🔄 Accessing ATLAS data with uproot...")
        
        with uproot.open(atlas_url, 
                        file_handler=uproot.source.xrootd.XRootDSource,
                        timeout=300, 
                        num_fails=8) as file:
            
            print("✅ XRootD connection successful!")
            
            # Basic file analysis
            keys = list(file.keys())
            print(f"📁 File contains {len(keys)} objects")
            
            if "CollectionTree" in file:
                tree = file["CollectionTree"]
                print(f"🌳 CollectionTree: {tree.num_entries:,} events")
                print(f"🌿 Branches: {len(tree.keys())}")
                
                # Example: Read data from a branch
                branches = list(tree.keys())
                print(f"\\n📊 Available branches (first 10):")
                for i, branch in enumerate(branches[:10], 1):
                    print(f"  {i:2d}. {branch}")
                
                # Example data reading (uncomment to use)
                # print("\\n🔬 Reading sample data...")
                # sample_data = tree[branches[0]].array(library="np", entry_stop=1000)
                # print(f"Sample data from '{branches[0]}': {len(sample_data)} entries")
                
                print("\\n🎯 Ready for physics analysis!")
                return tree  # Return tree for further analysis
                
    except Exception as e:
        print(f"❌ uproot method failed: {e}")
        return None
'''
    
    elif method == "ROOT":
        script_content += '''
    # Method: ROOT TFile with SSL bypass
    try:
        import ROOT
        
        print("🔄 Accessing ATLAS data with ROOT...")
        
        # Configure ROOT for network access
        ROOT.gEnv.SetValue("XNet.ConnectTimeout", 300)
        ROOT.gEnv.SetValue("XNet.RequestTimeout", 300)
        ROOT.gErrorIgnoreLevel = ROOT.kWarning
        
        tfile = ROOT.TFile.Open(atlas_url)
        
        if tfile and not tfile.IsZombie():
            print("✅ ROOT TFile connection successful!")
            
            # Basic file analysis
            keys = tfile.GetListOfKeys()
            print(f"📁 File contains {keys.GetSize()} objects")
            
            tree = tfile.Get("CollectionTree")
            if tree:
                print(f"🌳 CollectionTree: {tree.GetEntries():,} events")
                print(f"🌿 Branches: {tree.GetNbranches()}")
                
                # Show branch names
                print(f"\\n📊 Available branches (first 10):")
                branches = tree.GetListOfBranches()
                for i in range(min(10, branches.GetSize())):
                    branch = branches.At(i)
                    print(f"  {i+1:2d}. {branch.GetName()}")
                
                print("\\n🎯 Ready for ROOT-based analysis!")
                return tfile, tree  # Return for further analysis
        else:
            print("❌ Could not open ROOT file")
            return None
            
    except Exception as e:
        print(f"❌ ROOT method failed: {e}")
        return None
'''
    
    script_content += '''

if __name__ == "__main__":
    print("🚀 ATLAS XRootD Access with SSL Bypass")
    print("=" * 45)
    
    result = atlas_analysis()
    
    if result:
        print("\\n🎉 SUCCESS! ATLAS data is accessible via XRootD")
        print("\\n📝 You can now:")
        print("  1. Add your physics analysis code to this script")
        print("  2. Read branches and perform calculations")
        print("  3. Create histograms and plots")
    else:
        print("\\n❌ XRootD access failed")
        print("💡 Try the HTTP download method as backup")
'''
    
    filename = f"atlas_xrootd_{method.lower()}.py"
    with open(filename, 'w') as f:
        f.write(script_content)
    
    print(f"📝 Created '{filename}' - working XRootD script")

def create_bash_wrapper():
    """Create bash script with environment setup"""
    
    bash_content = '''#!/bin/bash
# ATLAS XRootD Access - Bash Wrapper
# Sets up environment and runs Python script

echo "🔧 Setting up XRootD SSL bypass environment"

# Export XRootD environment variables
export XRD_TLSNOCHECK=1
export XRD_TLSNOVERIFY=1
export XRD_CONNECTIONRETRY=5
export XRD_REQUESTTIMEOUT=300
export XRD_NETWORKSTACK=IPAuto

echo "✅ Environment configured"
echo "XRD_TLSNOCHECK=$XRD_TLSNOCHECK"
echo "XRD_TLSNOVERIFY=$XRD_TLSNOVERIFY"

# Run your Python script
echo -e "\\n🚀 Running ATLAS analysis..."

# Option 1: Run with uproot
if [ -f "atlas_xrootd_uproot.py" ]; then
    python3 atlas_xrootd_uproot.py
elif [ -f "atlas_xrootd_root.py" ]; then
    python3 atlas_xrootd_root.py
else
    echo "❌ No working script found"
    echo "💡 Run the SSL bypass test first"
fi
'''
    
    with open('run_atlas_xrootd.sh', 'w') as f:
        f.write(bash_content)
    
    os.chmod('run_atlas_xrootd.sh', 0o755)
    print("📝 Created 'run_atlas_xrootd.sh' - bash wrapper script")

def main():
    """Main testing function"""
    print("🚀 XRootD SSL Certificate Fix for ATLAS Data")
    print("=" * 50)
    
    # Set up environment
    setup_xrootd_ssl_bypass()
    
    working_methods = []
    
    # Test uproot
    if test_uproot_with_ssl_bypass():
        working_methods.append("uproot")
    
    # Test ROOT
    if test_root_with_ssl_bypass():
        working_methods.append("ROOT")
    
    # Results
    print_banner("Results Summary")
    
    if working_methods:
        print("🎉 SUCCESS! XRootD access working with SSL bypass:")
        for method in working_methods:
            print(f"  ✅ {method}")
        
        # Create working scripts
        for method in working_methods:
            create_working_script(method)
        
        create_bash_wrapper()
        
        print(f"\\n🎯 Next steps:")
        print("1. Use the generated scripts for your analysis")
        print("2. Add your physics analysis code")
        print("3. Run with: ./run_atlas_xrootd.sh")
        
    else:
        print("❌ SSL bypass didn't work")
        print("\\n💡 Alternative solutions:")
        print("1. Download file with wget --no-check-certificate")
        print("2. Use HTTP method with verify=False")
        print("3. Ask cluster admin to fix certificate configuration")

if __name__ == "__main__":
    main()