# Import basic libraries
import copy # copy variables
import os   # manage paths

import uproot   # use of root files
import awkward as ak    # nested, variable sized data
import vector   # lorentz vectors
vector.register_awkward()
import matplotlib.pyplot as plt # plotting
import tqdm # progress bars

filename = "Collision Data/mc20_13TeV_MC_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_file_index.json"
tree = uproot.open({filename: "CollectionTree"})

# el_pt = tree["AnalysisElectronsAuxDyn.pt"].array()

# el_pt_np = ak.flatten(el_pt).to_numpy()


electrons = ak.zip(
    {
        "pt": tree["AnalysisElectronsAuxDyn.pt"].array(),
        "eta": tree["AnalysisElectronsAuxDyn.eta"].array(),
        "phi": tree["AnalysisElectronsAuxDyn.phi"].array(),
    }
)

muons = ak.zip(
     {
        "pt": tree["AnalysisMuonsAuxDyn.pt"].array(),
        "eta": tree["AnalysisMuonsAuxDyn.eta"].array(),
        "phi": tree["AnalysisMuonsAuxDyn.phi"].array(),
    }
)

jets = ak.zip(
     {
        "pt": tree["AnalysisJetsAuxDyn.pt"].array(),
        "eta": tree["AnalysisJetsAuxDyn.eta"].array(),
        "phi": tree["AnalysisJetsAuxDyn.phi"].array(),
        "mass": tree["AnalysisJetsAuxDyn.m"].array(),
    }
)



####
####
####
# SCHEMAS
####
####
####
schema = {
    "Electrons": [
        "pt", "eta", "phi",
    ],
    "Muons": [
        "pt", "eta", "phi",
    ],
    "Jets": [
        "pt", "eta", "phi", "m"
    ],
    "BTagging_AntiKt4EMPFlow": [
        "DL1dv01_pb",
    ]
}


def read_events(filename, schema):
    with uproot.open({filename: "CollectionTree"}) as tree:
        events = {}
        for objname, fields in schema.items():
            base = objname
            if objname in ["Electrons", "Muons", "Jets"]:
                base = "Analysis" + objname
                ak_zip = vector.zip
            else:
                ak_zip = ak.zip
            arrays = tree.arrays(
                fields,
                aliases={field: f"{base}AuxDyn.{field}" for field in fields},
            )
            arrays = ak_zip(dict(zip(arrays.fields, ak.unzip(arrays))))
            events[objname] = arrays
        return ak.zip(events, depth_limit=1)
    
events = read_events(filename, schema)

print(events)