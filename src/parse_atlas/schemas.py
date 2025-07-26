PARTICLE_LIST = ['Electrons', 'Muons', 'Jets', 'Photons']

INVARIANT_MASS_SCHEMA = {
    "Electrons": [
        "pt",   # AnalysisElectronsAuxDyn.pt
        "eta",  # AnalysisElectronsAuxDyn.eta
        "phi",  # AnalysisElectronsAuxDyn.phi
        "m"     # AnalysisElectronsAuxDyn.m
    ],
    "Muons": [
        "pt",   # AnalysisMuonsAuxDyn.pt
        "eta",  # AnalysisMuonsAuxDyn.eta
        "phi"   # AnalysisMuonsAuxDyn.phi
        # 'm' missing from metadata; assume constant ~0.105 or drop invariant mass for muons
    ],
    "Jets": [
        "pt",   # AnalysisJetsAuxDyn.pt
        "eta",  # AnalysisJetsAuxDyn.eta
        "phi",  # AnalysisJetsAuxDyn.phi
        "m"     # AnalysisJetsAuxDyn.m
    ],
    "Photons": [
        "pt",   # if available from a similar PhotonsAuxDyn
        "eta",
        "phi"
        # usually massless, so 'm' can be assumed = 0 if needed
    ]
}