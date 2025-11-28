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
    ]
}

SCHEMA_RANDOM = {
    "electron": [
        "pt",  # AnalysisElectronsAuxDyn.rho  (transverse momentum)
        "eta",  # AnalysisElectronsAuxDyn.eta  (pseudorapidity)
        "phi",  # AnalysisElectronsAuxDyn.phi  (azimuthal angle)
        "mass"   # AnalysisElectronsAuxDyn.tau  (energy/time)
    ],
    "muon": [
        "pt",  # AnalysisElectronsAuxDyn.rho  (transverse momentum)
        "eta",  # AnalysisElectronsAuxDyn.eta  (pseudorapidity)
        "phi",  # AnalysisElectronsAuxDyn.phi  (azimuthal angle)
        "mass"   # AnalysisElectronsAuxDyn.tau  (energy/time)
    ],
    "jet": [
        "pt",  # AnalysisElectronsAuxDyn.rho  (transverse momentum)
        "eta",  # AnalysisElectronsAuxDyn.eta  (pseudorapidity)
        "phi",  # AnalysisElectronsAuxDyn.phi  (azimuthal angle)
        "mass"   # AnalysisElectronsAuxDyn.tau  (energy/time)
    ],
    "photon": [
        "pt",  # AnalysisPhotonsAuxDyn.rho
        "eta",  # AnalysisPhotonsAuxDyn.eta
        "phi"   # AnalysisPhotonsAuxDyn.phi
    ]
}
