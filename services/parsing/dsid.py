"""
Monte-Carlo dataset number (DSID) resolution.

The DSID keys every per-dataset normalization input (cross section, k-factor,
filter efficiency, sum of weights), so it must be identified reliably. It is
read only from the ``mcChannelNumber`` carried by the events themselves, never
inferred from file names or URLs.
"""

from typing import Optional

import awkward as ak
import numpy as np

from domain.events import MC_EVENT_INFO_FIELD, MC_CHANNEL_NUMBER_FIELD


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
