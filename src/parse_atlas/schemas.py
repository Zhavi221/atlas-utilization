PARTICLE_LIST = ['Electrons', 'Muons', 'Jets', 'Photons']

#IS THIS REQUIRED
INVARIANT_MASS_SCHEMA_ORIGINAL = {
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
INVARIANT_MASS_SCHEMA = {
    "Electrons": [
        "pt",  # AnalysisElectronsAuxDyn.rho  (transverse momentum)
        "eta",  # AnalysisElectronsAuxDyn.eta  (pseudorapidity)
        "phi",  # AnalysisElectronsAuxDyn.phi  (azimuthal angle)
        "mass"   # AnalysisElectronsAuxDyn.tau  (energy/time)
    ],
    "Muons": [
        "pt",  # AnalysisElectronsAuxDyn.rho  (transverse momentum)
        "eta",  # AnalysisElectronsAuxDyn.eta  (pseudorapidity)
        "phi",  # AnalysisElectronsAuxDyn.phi  (azimuthal angle)
        "mass"   # AnalysisElectronsAuxDyn.tau  (energy/time)
    ],
    "Jets": [
        "pt",  # AnalysisElectronsAuxDyn.rho  (transverse momentum)
        "eta",  # AnalysisElectronsAuxDyn.eta  (pseudorapidity)
        "phi",  # AnalysisElectronsAuxDyn.phi  (azimuthal angle)
        "mass"   # AnalysisElectronsAuxDyn.tau  (energy/time)
    ],
    "Photons": [
        "pt",  # AnalysisPhotonsAuxDyn.rho
        "eta",  # AnalysisPhotonsAuxDyn.eta
        "phi"   # AnalysisPhotonsAuxDyn.phi
        # No tau field listed for photons; usually treated as massless
    ]
}
