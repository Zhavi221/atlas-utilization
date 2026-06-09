"""
MetadataFetcher service - Fetches file metadata from ATLAS Open Data API.

Single responsibility: Interact with ATLAS API to get file URLs.
"""

import logging
import json
import re
import requests
import io
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Optional
import atlasopenmagic as atom

from services import consts
from domain.metadata import ReleaseMetadata

# Matches the dataset namespace component in ATLAS Open Data URLs.
# The digit+underscore suffix (e.g. mc20_, data16_) is specific enough
# to avoid false matches like "opendata" or "metadata".
# Examples:
#   MC:   .../mc20_13TeV/DAOD_PHYSLITE.xxx
#   Data: .../data16_13TeV/DAOD_PHYSLITE.xxx


class UrlType(Enum):
    DATA    = "data"
    MC      = "mc"


_MC_NAMESPACE_RE   = re.compile(r'/mc\d+_',   re.IGNORECASE)
_DATA_NAMESPACE_RE = re.compile(r'/data\d+_', re.IGNORECASE)


def _classify_url(url: str) -> UrlType:
    """
    Classify a URL as DATA or MC using anchored regex on the RUCIO namespace.

    Raises ValueError if the URL matches both patterns or neither — so
    unclassifiable URLs are never silently misrouted.
    """
    is_mc   = bool(_MC_NAMESPACE_RE.search(url))
    is_data = bool(_DATA_NAMESPACE_RE.search(url))

    if is_mc and not is_data:
        return UrlType.MC
    if is_data and not is_mc:
        return UrlType.DATA
    raise ValueError(
        f"URL matched {'both' if is_mc and is_data else 'neither'} "
        f"MC and DATA patterns — cannot classify: {url}"
    )


def _assert_no_cross_contamination(separated: dict[str, list[str]]) -> None:
    """
    Post-separation integrity check.
    Asserts no MC URLs ended up in data keys and vice versa.
    Raises AssertionError immediately if violated.
    """
    for key, urls in separated.items():
        is_mc_key = key.endswith("_mc")
        for url in urls:
            url_type = _classify_url(url)
            if is_mc_key and url_type != UrlType.MC:
                raise AssertionError(
                    f"DATA url found in MC key '{key}': {url}"
                )
            if not is_mc_key and url_type != UrlType.DATA:
                raise AssertionError(
                    f"MC url found in DATA key '{key}': {url}"
                )
# ── END CHANGE 1 ─────────────────────────────────────────────────────────────


class MetadataFetcher:
    """
    Service for fetching file metadata from ATLAS Open Data API.
    
    Uses atlasopenmagic library to interact with CERN Open Data Portal.
    """
    
    def __init__(self, timeout: int = 60, show_progress: bool = False):
        """
        Initialize metadata fetcher.
        
        Args:
            timeout: Timeout for API requests in seconds
            show_progress: Whether to show progress messages
        """
        self.timeout = timeout
        self.show_progress = show_progress
        self._available_releases = None
    
    def get_available_releases(self) -> dict:
        """
        Get all available release years from ATLAS Open Data.
        
        Returns:
            Dict mapping release names to their metadata
        """
        if self._available_releases is None:
            if self.show_progress:
                self._available_releases = atom.available_releases()
            else:
                # Suppress output
                with redirect_stdout(io.StringIO()):
                    self._available_releases = atom.available_releases()
        
        return self._available_releases
    
    def validate_release_years(self, release_years: list[str]) -> None:
        """
        Validate that release years are available.
        
        Args:
            release_years: List of release year strings to validate
            
        Raises:
            ValueError: If any release year is not available
        """
        if not release_years:
            return
        
        available = self.get_available_releases()
        invalid = [year for year in release_years if year not in available]
        
        if invalid:
            available_list = list(available.keys())
            raise ValueError(
                f"Release years {invalid} are not recognized. "
                f"Available releases: {available_list}"
            )
    
    def fetch_by_release_years(
        self,
        release_years: list[str],
      # ── CHANGE 2: Remove `separate_mc` parameter entirely ────────────────
        # BEFORE: separate_mc: bool = False
        # AFTER:  parameter removed — separation is always performed.
        # There is no valid use case for mixing data and MC in the cache.
    ) -> dict[str, list[str]]:
        """
        Fetch file URLs for given release years.
        
        Args:
            release_years: List of release years to fetch
            
        Returns:
            Dict mapping release years to lists of file URLs
        """
        self.validate_release_years(release_years)
        
        release_files = self._fetch_urls_for_releases(release_years)
        
        # ── CHANGE 3: Always separate, remove `if separate_mc` branch ────────
        # BEFORE:
        #     if separate_mc:
        #         release_files = self._separate_mc_files(release_files)
        #     return release_files
        #
        # AFTER: always separate, using strict classifier:
        return self._separate_mc_files(release_files)
        # ── END CHANGE 3 ─────────────────────────────────────────────────────
    
    def fetch_by_record_ids(
        self,
        record_ids: list[int]
    ) -> dict[str, list[str]]:
        """
        Fetch file URLs from specific CMS record IDs.
        
        Args:
            record_ids: List of CMS record IDs to fetch
            
        Returns:
            Dict mapping record keys to lists of file URLs
            Keys are formatted as "record_{id}"
        """
        release_files = {}
        
        for record_id in record_ids:
            try:
                file_urls = self._fetch_files_for_record(record_id)

                release_files[f"record_{record_id}"] = file_urls
                
                logging.info(f"Fetched {len(file_urls)} files from record {record_id}")
            except Exception as e:
                logging.warning(f"Failed to fetch record {record_id}: {e}")
        
        return release_files
    
    def fetch(
        self,
        release_years: Optional[list[str]] = None,
        record_ids: Optional[list[int]] = None,
         # ── CHANGE 4: Remove `separate_mc` parameter from public API ─────────
        # BEFORE: separate_mc: bool = False
        # AFTER:  parameter removed — always separated.
        # ── END CHANGE 4 ─────
    ) -> dict[str, list[str]]:
        """
        Fetch file URLs using either release years or specific record IDs.
        
        Priority: record_ids > release_years > all available releases
        
        Args:
            release_years: Optional list of release years
            record_ids: Optional list of specific record IDs
            
        Returns:
            Dict mapping release/record keys to lists of file URLs
        """
        # Priority 1: Specific record IDs
        if record_ids and len(record_ids) > 0:
            logging.info(f"Fetching URIs from specific_record_ids: {record_ids}")
            return self.fetch_by_record_ids(record_ids)
        
        # Priority 2: Specified release years
        if release_years and len(release_years) > 0:
            return self.fetch_by_release_years(release_years)
        
        # Priority 3: All available releases
        available = self.get_available_releases()
        all_releases = list(available.keys())
        logging.info(f"No specific releases specified, fetching all: {all_releases}")
        return self.fetch_by_release_years(all_releases)
    
    def _fetch_urls_for_releases(self, release_years: list[str]) -> dict[str, list[str]]:
        """
        Fetch file URLs for multiple release years using thread pool.
        
        Args:
            release_years: List of release years
            
        Returns:
            Dict mapping release years to file URL lists
        """
        release_files = {}
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            for year in release_years:
                if year not in release_files:
                    release_files[year] = []
                
                try:
                    # Set the release in atlasopenmagic
                    future = executor.submit(atom.set_release, year)
                    future.result(timeout=self.timeout)
                    
                    # Get available datasets
                    datasets = atom.available_datasets()
                    
                    # Fetch URLs for each dataset
                    for dataset_id in datasets:
                        future = executor.submit(atom.get_urls, dataset_id)
                        urls = future.result(timeout=self.timeout)
                        if urls:
                            release_files[year].extend(urls)
                
                except TimeoutError:
                    logging.warning(
                        f"Timeout while fetching metadata for release year {year}"
                    )
                except Exception as e:
                    logging.warning(
                        f"Could not fetch metadata for release year {year}: {e}"
                    )
        
        return release_files
    
    def _fetch_files_for_record(self, record_id: int) -> list[str]:
        """
        Fetch file URLs for a specific CMS record ID.
        
        Args:
            record_id: CMS record ID
            
        Returns:
            List of file URLs
        """
        url = consts.CMS_RECID_FILEPAGE_URL.format(record_id)
        response = requests.get(url)
        response.raise_for_status()
        
        data = json.loads(response.text)
        file_list = []
        
        for index_file in data["index_files"]["files"]:
            for file_entry in index_file["files"]:
                file_uri = file_entry["uri"]
                file_list.append(file_uri)
        
        return file_list
    

    @staticmethod
    def _separate_mc_files(
        release_files: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        # ── CHANGE 5: Replace loose substring match with strict classifier ────
        # BEFORE (broken):
        #     for url in file_urls:
        #         if "mc" in url.lower():      # hits "opendata" in ALL URLs
        #             mc_files.append(url)
        #         else:
        #             data_files.append(url)   # unclassifiable silently go here
        #
        # AFTER (strict):
        separated: dict[str, list[str]] = {}
        classification_errors: list[str] = []

        for year, file_urls in release_files.items():
            data_files: list[str] = []
            mc_files:   list[str] = []

            for url in file_urls:
                try:
                    url_type = _classify_url(url)
                except ValueError as e:
                    classification_errors.append(str(e))
                    continue

                if url_type == UrlType.MC:
                    mc_files.append(url)
                else:
                    data_files.append(url)

            if classification_errors:
                raise ValueError(
                    f"Failed to classify {len(classification_errors)} URL(s):\n"
                    + "\n".join(classification_errors[:10])
                    + (f"\n... and {len(classification_errors) - 10} more"
                       if len(classification_errors) > 10 else "")
                )

            if data_files:
                separated[year] = data_files
            if mc_files:
                separated[f"{year}_mc"] = mc_files

            logging.info(
                f"Release '{year}': {len(data_files)} data files, "
                f"{len(mc_files)} MC files"
            )

        # Post-separation integrity check — asserts zero cross-contamination
        _assert_no_cross_contamination(separated)
        # ── END CHANGE 5 ─────────────────────────────────────────────────────

        return separated
    
    def to_release_metadata(
        self,
        release_files: dict[str, list[str]]
    ) -> dict[str, ReleaseMetadata]:
        """
        Convert file URLs dict to ReleaseMetadata objects.
        
        Args:
            release_files: Dict mapping release keys to file URL lists
            
        Returns:
            Dict mapping release keys to ReleaseMetadata objects
        """
        metadata_dict = {}
        
        for release_key, file_urls in release_files.items():
            # Extract record IDs from URLs (assuming they're numeric)
            # For now, use URL hash as file ID
            file_ids = [hash(url) for url in file_urls]
            
            metadata = ReleaseMetadata.from_file_list(release_key, file_ids)
            metadata_dict[release_key] = metadata
        
        return metadata_dict
