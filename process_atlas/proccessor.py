from typing import Dict, Any, Optional
import awkward as ak
import vector
import logging
import copy
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants (can be moved to a config file or passed as parameters)
DEFAULT_PT_CUT = 10  # in GeV
DEFAULT_ETA_CUT_MUONS = 2.5
DEFAULT_ETA_CUT_ELECTRONS = 2.47

class ATLASProcessor:
    """
    A class to process ATLAS event data, including particle selection,
    event filtering, and invariant mass calculation.
    """

    def __init__(self, events: ak.Array, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the processor with event data and optional configuration.

        Args:
            events (ak.Array): The event data in an Awkward Array.
            config (Optional[Dict[str, Any]]): Configuration dictionary for cuts and thresholds.
        """
        self.events = events
        self.config = config or {}
        self._validate_events()

    def _validate_events(self) -> None:
        """
        Validate the event data to ensure it contains required fields.
        """
        required_fields = ["Electrons", "Muons"]
        for field in required_fields:
            if field not in self.events.fields:
                raise ValueError(f"Input events must contain '{field}' field.")

    def _apply_pt_eta_cuts(self, particles: ak.Array, pt_cut: float, eta_cut: float) -> ak.Array:
        """
        Apply transverse momentum (pt) and pseudorapidity (eta) cuts to particles.

        Args:
            particles (ak.Array): The particle data.
            pt_cut (float): Minimum pt threshold.
            eta_cut (float): Maximum |eta| threshold.

        Returns:
            ak.Array: Filtered particles.
        """
        return particles[(particles.pt > pt_cut) & (abs(particles.eta) < eta_cut)]

    def select_muons(self) -> ak.Array:
        """
        Select muons based on pt and eta cuts.

        Returns:
            ak.Array: Selected muons.
        """
        pt_cut = self.config.get("muon_pt_cut", DEFAULT_PT_CUT)
        eta_cut = self.config.get("muon_eta_cut", DEFAULT_ETA_CUT_MUONS)
        return self._apply_pt_eta_cuts(self.events.Muons, pt_cut, eta_cut)

    def select_electrons(self) -> ak.Array:
        """
        Select electrons based on pt and eta cuts.

        Returns:
            ak.Array: Selected electrons.
        """
        pt_cut = self.config.get("electron_pt_cut", DEFAULT_PT_CUT)
        eta_cut = self.config.get("electron_eta_cut", DEFAULT_ETA_CUT_ELECTRONS)
        return self._apply_pt_eta_cuts(self.events.Electrons, pt_cut, eta_cut)

    def filter_events(self) -> ak.Array:
        """
        Filter events to select those with exactly two electrons, one of which is negative.

        Returns:
            ak.Array: Filtered events.
        """
        events = copy.copy(self.events)  # Shallow copy to avoid modifying the original
        events["Electrons"] = self.select_electrons()
        events["Muons"] = self.select_muons()

        # Add charge information
        events["Electrons", "is_neg"] = events.Electrons.charge < 0
        events["Muons", "is_neg"] = events.Muons.charge < 0

        # Apply event selection criteria
        filtered_events = events[
            (ak.num(events.Electrons) == 2)  # Exactly two electrons
            & (ak.num(events.Electrons[events.Electrons.is_neg]) == 1)  # One negative electron
        ]
        return ak.to_packed(filtered_events)

    def calculate_invariant_mass(self, particles: ak.Array) -> ak.Array:
        """
        Calculate the invariant mass of particle pairs.

        Args:
            particles (ak.Array): The particle data.

        Returns:
            ak.Array: Invariant mass of particle pairs.
        """
        particle_pairs = ak.combinations(particles, 2)
        p1, p2 = ak.unzip(particle_pairs)
        combined_p4 = p1 + p2
        return combined_p4.to_ptphietamass().mass

    def process(self) -> None:
        """
        Process the events: filter, calculate invariant mass, and plot the results.
        """
        try:
            filtered_events = self.filter_events()
            logger.info(f"Total events: {len(self.events):,}")
            logger.info(f"Events after filtering: {len(filtered_events)}")

            if len(filtered_events) == 0:
                logger.warning("No events passed the selection criteria.")
                return

            # Calculate invariant mass for electron pairs
            inv_mass = self.calculate_invariant_mass(filtered_events.Electrons)

            # Plot the invariant mass distribution
            plt.hist(inv_mass, bins=100, range=(0, 500), label="Electron Pairs")
            plt.xlabel("Invariant Mass (GeV)")
            plt.ylabel("Number of Events")
            plt.title("Invariant Mass Distribution of Electron Pairs")
            plt.legend()
            plt.show()

        except Exception as e:
            logger.error(f"An error occurred during processing: {e}")
            raise

# Example usage:
# Assuming `events` is an Awkward Array loaded with ATLAS event data
# config = {
#     "muon_pt_cut": 10,
#     "muon_eta_cut": 2.5,
#     "electron_pt_cut": 10,
#     "electron_eta_cut": 2.47,
# }
# processor = ATLASProcessor(events, config)
# processor.process()