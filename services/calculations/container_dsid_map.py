"""
Container-ID -> physics-DSID resolution for ATLAS Open Data.

Why this exists:
    ATLAS Open Data XRootD URLs identify a file by its *rucio container ID*
    (an 8-digit number, e.g. ``.../DAOD_PHYSLITE.37620644._000001.pool.root.1``),
    NOT by the 6-digit physics DSID (e.g. ``410470``) that dataset metadata is
    keyed on. One physics DSID typically maps to 1-4 container IDs. The strict
    regex in :func:`services.calculations.weights_registry.extract_dsid_from_url`
    only recognises the *canonical* 6-digit DSID token, so it returns ``None``
    for a raw Open Data URL and no automatic source->DSID mapping is built.

What this does:
    Builds a ``{container_id: physics_dsid}`` reverse lookup by asking
    ``atlasopenmagic`` for every dataset's file URLs and extracting the
    container IDs from them. The map is cached to JSON so the (network-heavy)
    build happens once. :meth:`ContainerDsidResolver.resolve` first tries the
    canonical regex, then falls back to the container lookup.

Kept separate from ``weights_registry.extract_dsid_from_url`` on purpose:
    that function stays pure and offline (no network, deterministic) so the
    smoke test can assert "no false positives". Container resolution is the
    stateful, network-backed layer on top.
"""

import json
import logging
import os
import re
from typing import Iterable, List, Optional

from services.calculations.weights_registry import extract_dsid_from_url

logger = logging.getLogger(__name__)

# An 8-digit token that is not part of a longer digit run. Matches a rucio
# container ID (37620644) but NOT a 6-digit file-sequence index (_000001) or a
# 6-digit physics DSID (410470).
_CONTAINER_ID_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")

_CACHE_RELEASES_KEY = "_releases_built"
_CACHE_MAP_KEY = "map"


def extract_container_ids(url_or_name: str) -> List[str]:
    """Return every 8-digit container-ID-shaped token in a URL or filename."""
    if not url_or_name:
        return []
    return _CONTAINER_ID_RE.findall(url_or_name)


class ContainerDsidResolver:
    """
    Resolve a source URL/filename to a physics DSID, container-ID aware.

    Resolution order for :meth:`resolve`:
        1. Canonical 6-digit DSID via :func:`extract_dsid_from_url` (offline).
        2. 8-digit container ID looked up in the reverse map (built lazily from
           atlasopenmagic and cached to ``cache_path``).

    The reverse map is only built when a container lookup is actually needed,
    so pipelines whose filenames already carry a canonical DSID never pay the
    network cost.
    """

    def __init__(
        self,
        releases: Optional[Iterable[str]] = None,
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            releases: Open Data release tags to build the map from (e.g.
                ``["2024r-pp"]``). When None, all releases reported by
                atlasopenmagic are scanned (slower first build).
            cache_path: Where to persist the reverse map JSON. When None,
                the map is held in memory only and rebuilt each process.
        """
        # None => scan all releases (needs the network); an explicit (possibly
        # empty) iterable is honoured as-is, so releases=[] builds nothing.
        self._releases = list(releases) if releases is not None else None
        self._cache_path = cache_path
        self._map: Optional[dict] = None
        self._releases_built: set = set()

    # -- public API ----------------------------------------------------- #

    def resolve(self, url_or_name: str) -> Optional[int]:
        """Resolve a URL/filename to a physics DSID, or None if unresolved."""
        dsid = extract_dsid_from_url(url_or_name)
        if dsid is not None:
            return dsid

        containers = extract_container_ids(url_or_name)
        if not containers:
            return None

        reverse = self._ensure_map()
        for container_id in containers:
            mapped = reverse.get(container_id)
            if mapped is not None:
                return int(mapped)
        return None

    def resolve_many(self, urls: Iterable[str]) -> dict:
        """Resolve a batch of URLs to ``{url: dsid}`` (None values dropped)."""
        out = {}
        for url in urls:
            dsid = self.resolve(url)
            if dsid is not None:
                out[url] = dsid
        return out

    # -- map construction ----------------------------------------------- #

    def _ensure_map(self) -> dict:
        """Return the reverse map, loading cache / building it if needed."""
        if self._map is not None:
            return self._map

        self._map = {}
        self._load_cache()

        wanted = self._releases if self._releases is not None else self._available_releases()
        missing = [r for r in wanted if r not in self._releases_built]
        if missing:
            self._build_for_releases(missing)
            self._save_cache()

        return self._map

    def _load_cache(self) -> None:
        if not self._cache_path or not os.path.exists(self._cache_path):
            return
        try:
            with open(self._cache_path, "r") as f:
                data = json.load(f)
            self._map = dict(data.get(_CACHE_MAP_KEY, {}))
            self._releases_built = set(data.get(_CACHE_RELEASES_KEY, []))
            logger.info(
                "Loaded container->DSID cache: %d containers, releases=%s",
                len(self._map), sorted(self._releases_built),
            )
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Could not load container->DSID cache %s: %s", self._cache_path, e)
            self._map = {}
            self._releases_built = set()

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._cache_path)), exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(
                    {
                        _CACHE_MAP_KEY: self._map,
                        _CACHE_RELEASES_KEY: sorted(self._releases_built),
                    },
                    f, indent=2,
                )
        except IOError as e:
            logger.warning("Could not write container->DSID cache %s: %s", self._cache_path, e)

    @staticmethod
    def _available_releases() -> List[str]:
        import atlasopenmagic as atom
        try:
            # available_releases() returns a {release: ...} mapping; keys are the tags.
            return list(atom.available_releases())
        except Exception as e:
            logger.warning("Could not list atlasopenmagic releases: %s", e)
            return []

    def _build_for_releases(self, releases: Iterable[str]) -> None:
        """Populate the reverse map for the given releases via atlasopenmagic."""
        import atlasopenmagic as atom

        for release in releases:
            added = 0
            try:
                atom.set_release(release)
                datasets = atom.available_datasets()
            except Exception as e:
                logger.warning("Could not enumerate datasets for release %s: %s", release, e)
                self._releases_built.add(release)  # don't retry a broken release every call
                continue

            for dataset_id in datasets:
                dsid = _to_int(dataset_id)
                if dsid is None:
                    continue
                try:
                    urls = atom.get_urls(dataset_id) or []
                except Exception as e:
                    logger.debug("get_urls failed for %s: %s", dataset_id, e)
                    continue
                for url in urls:
                    for container_id in extract_container_ids(url):
                        # First writer wins; warn only on a genuine conflict.
                        existing = self._map.get(container_id)
                        if existing is not None and int(existing) != dsid:
                            logger.warning(
                                "Container %s maps to both DSID %s and %s; keeping %s",
                                container_id, existing, dsid, existing,
                            )
                            continue
                        if existing is None:
                            self._map[container_id] = dsid
                            added += 1

            self._releases_built.add(release)
            logger.info("Built container->DSID for release %s: +%d containers", release, added)


def _to_int(value) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None
