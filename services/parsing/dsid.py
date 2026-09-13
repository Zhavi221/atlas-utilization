"""
Monte-Carlo dataset number (DSID) resolution.

The DSID keys every per-dataset normalization input (cross section, k-factor,
filter efficiency, sum of weights), so it must be identified reliably. The
authoritative source is the ``mcChannelNumber`` carried by the events
themselves; names are a fallback for files without that branch. A *wrong* DSID
silently produces a wrong weight — worse than an unweighted histogram — so name
matching is deliberately strict and returns ``None`` when in doubt.
"""

import re
from typing import Optional

import awkward as ak
import numpy as np

from domain.events import MC_EVENT_INFO_FIELD, MC_CHANNEL_NUMBER_FIELD

# Match the 6-digit ATLAS DSID only in its canonical position, e.g.
# "mc20_13TeV.410470.PhPy8EG_..." -> "410470". Rucio container numbers
# (DAOD_PHYSLITE.37620644._000001) and file-sequence indices must not match.
_DSID_PATTERNS = (
    re.compile(r"dsid_?(\d{6,8})"),            # dsid410470 / dsid_410470 (our chunk filenames)
    re.compile(r"[Tt]e[Vv]\.(\d{6})\."),      # ...TeV.410470.  (canonical)
    re.compile(r"\.(\d{6})\.[A-Za-z]"),        # .410470.PhPy8... (DSID before physics_short)
)


def extract_dsid_from_url(url_or_name: Optional[str]) -> Optional[int]:
    """
    Extract the dataset number (DSID) from an ATLAS Open Data URL or filename.

    Returns the DSID as an int, or None if no DSID-like token is found.
    """
    if not url_or_name:
        return None
    for pattern in _DSID_PATTERNS:
        match = pattern.search(url_or_name)
        if match:
            return int(match.group(1))
    return None


def dsids_in_events(events: ak.Array) -> np.ndarray:
    """Distinct dataset numbers carried by ``events`` (empty for data / no MC info)."""
    if len(events) == 0 or MC_EVENT_INFO_FIELD not in events.fields:
        return np.array([], dtype=np.int64)
    info = events[MC_EVENT_INFO_FIELD]
    if MC_CHANNEL_NUMBER_FIELD not in info.fields:
        return np.array([], dtype=np.int64)
    channels = np.unique(ak.to_numpy(info[MC_CHANNEL_NUMBER_FIELD]))
    return channels[channels > 0]  # 0 marks events from files without the branch


def dsid_of_events(events: ak.Array) -> Optional[int]:
    """The single DSID of ``events``, or None when unknown or mixed."""
    dsids = dsids_in_events(events)
    if len(dsids) == 1:
        return int(dsids[0])
    return None
