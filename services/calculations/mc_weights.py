"""
Monte-Carlo event-weight calculations.

Implements the standard per-event MC weight used to normalize simulated
samples to a target integrated luminosity:

    w = (sigma * k * eps_filter * L) / N_gen

where:
    sigma      = generator cross section
    k          = k-factor (higher-order correction)
    eps_filter = generator filter efficiency
    L          = target integrated luminosity
    N_gen      = effective number of generated events

In practice ``N_gen`` is the sum of per-event generator weights
(``sumOfWeights``), not the raw event count, because NLO generators can
produce events with weight != 1 (including negative weights). The two are
equal only for leading-order, unit-weight samples.

Units: cross section is given in picobarns (pb) and luminosity in inverse
femtobarns (fb^-1); we convert pb -> fb with ``PB_TO_FB`` so the product is
dimensionless (an event count).
"""

from domain.metadata import MCDatasetMetadata

# 1 pb = 1000 fb, so sigma[pb] * PB_TO_FB gives sigma in fb, matching L[fb^-1].
PB_TO_FB = 1000.0


def compute_normalization(
    metadata: MCDatasetMetadata,
    target_luminosity_fb: float,
) -> float:
    """
    Compute the per-dataset normalization factor.

    This is the weight every generated event would receive if each event had
    a generator weight of exactly 1. For weighted samples, multiply this by
    the per-event generator weight.

    Args:
        metadata: MC dataset metadata (cross section, k-factor, filter
            efficiency, sum of weights).
        target_luminosity_fb: Target integrated luminosity in fb^-1.

    Returns:
        Normalization factor: expected yield divided by the sum of weights.
    """
    if target_luminosity_fb < 0:
        raise ValueError(f"target_luminosity_fb must be non-negative, got {target_luminosity_fb}")

    expected_yield = (
        metadata.cross_section_pb
        * PB_TO_FB
        * metadata.k_factor
        * metadata.gen_filt_eff
        * target_luminosity_fb
    )
    return expected_yield / metadata.sum_of_weights

