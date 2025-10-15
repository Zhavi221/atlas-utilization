PARTICLE_LIST = ['Electrons', 'Muons', 'Jets', 'Photons']

INVARIANT_MASS_SCHEMA = {
    "Electrons": [
        "pt",   # AnalysisElectronsAuxDyn.pt
        "eta",  # AnalysisElectronsAuxDyn.eta
        "phi",  # AnalysisElectronsAuxDyn.phi
        "m",
        "rho",
        "tau"     # AnalysisElectronsAuxDyn.m
    ],
    "Muons": [
        "pt",   # AnalysisElectronsAuxDyn.pt
        "eta",  # AnalysisElectronsAuxDyn.eta
        "phi",  # AnalysisElectronsAuxDyn.phi
        "m",
        "rho",
        "tau"     # AnalysisElectronsAuxDyn.m
    ],
    "Jets": [
        "pt",   # AnalysisElectronsAuxDyn.pt
        "eta",  # AnalysisElectronsAuxDyn.eta
        "phi",  # AnalysisElectronsAuxDyn.phi
        "m",
        "rho",
        "tau"     # AnalysisElectronsAuxDyn.m
    ],
    "Photons": [
        "pt",   # AnalysisElectronsAuxDyn.pt
        "eta",  # AnalysisElectronsAuxDyn.eta
        "phi",  # AnalysisElectronsAuxDyn.phi
        "rho"
    ]
}
INVARIANT_MASS_SCHEMAa = {
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
