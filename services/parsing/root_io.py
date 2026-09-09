"""ROOT file opening helpers."""

from __future__ import annotations

from os import PathLike
from typing import Any

import uproot


def open_root_file(file_path: str | PathLike[str], **options: Any):
    """Open a ROOT file with a deterministic backend for XRootD URLs.

    Uproot's default fsspec XRootD backend maintains a background file-handle
    cache. Its cache pruner can race with active/closed handles and emit an
    unhandled ``Invalid operation`` exception. The native Uproot XRootD source
    owns and closes its resources with the ROOT file context instead.
    """
    if isinstance(file_path, str) and file_path.lower().startswith("root://"):
        from uproot.source.xrootd import XRootDSource

        options.setdefault("handler", XRootDSource)
    return uproot.open(file_path, **options)
