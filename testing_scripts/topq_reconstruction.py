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
btag_prob = tree["BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pb"].array()
jets["btag_prob"] = btag_prob
 
events = ak.zip({"Electrons": electrons, "Muons": muons, "Jets": jets}, depth_limit=1)

GeV = 1000.

def selected_electrons(el):
    return el[(el.pt > 30 * GeV) & (abs(el.eta) < 2.47)]

def selected_muons(mu):
    return mu[(mu.pt > 30 * GeV) & (abs(mu.eta) < 2.47)]

def selected_jets(j):
    return j[(j.pt > 30 * GeV) & (abs(j.eta) < 2.47)]

events["Electrons"] = selected_electrons(electrons)
events["Muons"] = selected_muons(muons)
events["Jets"] = selected_jets(jets)

events["Electrons"] = vector.awk(events.Electrons)
events["Muons"] = vector.awk(events.Muons)
events["Jets"] = vector.awk(events.Jets)

# jj, ee = ak.unzip(ak.cartesian([events.Jets, events.Electrons], nested=True))
# plt.hist(ak.flatten(jj.deltaR(ee), axis=None).to_numpy(), bins=100)
# plt.xlabel("$\Delta R(j, e)$ (for all jet - electron pairs)")
# plt.ylabel("Count of Jet-Electron Pairs")
# plt.title('Distribution of $\Delta R$ Between Jets and Electrons')
# plt.show()

def no_overlap(obj1, obj2, deltaR=0.4):
    obj1, obj2 = ak.unzip(ak.cartesian([obj1, obj2], nested=True))
    return ak.all(obj1.deltaR(obj2) > deltaR, axis=-1)

events["Jets"] = events.Jets[no_overlap(events.Jets, events.Electrons)]

events["Jets", "is_bjet"] = events.Jets.btag_prob > 0.85

events = events[
    (ak.num(events.Jets) >= 4) # at least 4 jets
    & ((ak.num(events.Electrons) + ak.num(events.Muons)) == 1) # exactly one lepton
    & (ak.num(events.Jets[events.Jets.is_bjet]) >= 2) # at least two btagged jets with prob > 0.85
]


def mjjj(jets):
    candidates = ak.combinations(jets, 3)
    j1, j2, j3 = ak.unzip(candidates)
    has_b = (j1.is_bjet + j2.is_bjet + j3.is_bjet) > 0
    candidates["p4"] = j1 + j2 + j3
    candidates = candidates[has_b]
    candidates = candidates[ak.argmax(candidates.p4.pt, axis=1, keepdims=True)]
    return candidates.p4.mass


plt.hist(ak.flatten(mjjj(events.Jets) / GeV, axis=None), bins=100)
plt.xlabel("Reconstructed Top Quark Mass (GeV)")
plt.ylabel("Number of Events")
plt.title("Distribution of Reconstructed Top Quark Mass")
plt.axvline(172.76, color='r', linestyle='dashed', linewidth=2, label='Expected Top Quark Mass')
plt.legend()
plt.show()
