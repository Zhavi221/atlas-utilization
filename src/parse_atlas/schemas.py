PARTICLE_LIST = ['Electrons', 'Muons', 'Jets', 'Photons']

INVARIANT_MASS_SCHEMAa = {
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
INVARIANT_MASS_SCHEMA = {
    "Electrons": [
        "rho",  # AnalysisElectronsAuxDyn.rho  (transverse momentum)
        "eta",  # AnalysisElectronsAuxDyn.eta  (pseudorapidity)
        "phi",  # AnalysisElectronsAuxDyn.phi  (azimuthal angle)
        "tau"   # AnalysisElectronsAuxDyn.tau  (energy/time)
    ],
    "Muons": [
        "rho",  # AnalysisMuonsAuxDyn.rho
        "eta",  # AnalysisMuonsAuxDyn.eta
        "phi",  # AnalysisMuonsAuxDyn.phi
        "tau"   # AnalysisMuonsAuxDyn.tau
    ],
    "Jets": [
        "rho",  # AnalysisJetsAuxDyn.rho
        "eta",  # AnalysisJetsAuxDyn.eta
        "phi",  # AnalysisJetsAuxDyn.phi
        "tau"   # AnalysisJetsAuxDyn.tau
    ],
    "Photons": [
        "rho",  # AnalysisPhotonsAuxDyn.rho
        "eta",  # AnalysisPhotonsAuxDyn.eta
        "phi"   # AnalysisPhotonsAuxDyn.phi
        # No tau field listed for photons; usually treated as massless
    ]
}
