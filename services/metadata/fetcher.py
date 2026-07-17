"""
MetadataFetcher service - Fetches file metadata from ATLAS Open Data API.

Single responsibility: Interact with ATLAS API to get file URLs.
"""

import logging
import json
import requests
import io
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import atlasopenmagic as atom

from services import consts
from domain.metadata import ReleaseMetadata, MCDatasetMetadata


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
        separate_mc: bool = False
    ) -> dict[str, list[str]]:
        """
        Fetch file URLs for given release years.
        
        Args:
            release_years: List of release years to fetch
            separate_mc: Whether to separate MC files from data files
            
        Returns:
            Dict mapping release years to lists of file URLs
        """
        self.validate_release_years(release_years)
        
        release_files = self._fetch_urls_for_releases(release_years)
        
        if separate_mc:
            release_files = self._separate_mc_files(release_files)
        
        return release_files
    
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
                record_key = f"record_{record_id}"
                release_files[record_key] = file_urls
                
                logging.info(f"Fetched {len(file_urls)} files from record {record_id}")
            except Exception as e:
                logging.warning(f"Failed to fetch record {record_id}: {e}")
        
        return release_files
    
    def fetch(
        self,
        release_years: Optional[list[str]] = None,
        record_ids: Optional[list[int]] = None,
        separate_mc: bool = False
    ) -> dict[str, list[str]]:
        """
        Fetch file URLs using either release years or specific record IDs.
        
        Priority: record_ids > release_years > all available releases
        
        Args:
            release_years: Optional list of release years
            record_ids: Optional list of specific record IDs
            separate_mc: Whether to separate MC files from data files
            
        Returns:
            Dict mapping release/record keys to lists of file URLs
        """
        # Priority 1: Specific record IDs
        if record_ids and len(record_ids) > 0:
            logging.info(f"Fetching URIs from specific_record_ids: {record_ids}")
            return self.fetch_by_record_ids(record_ids)
        
        # Priority 2: Specified release years
        if release_years and len(release_years) > 0:
            return self.fetch_by_release_years(release_years, separate_mc)
        
        # Priority 3: All available releases
        available = self.get_available_releases()
        all_releases = list(available.keys())
        logging.info(f"No specific releases specified, fetching all: {all_releases}")
        return self.fetch_by_release_years(all_releases, separate_mc)
    
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
    def _separate_mc_files(release_files: dict[str, list[str]]) -> dict[str, list[str]]:
        """
        Separate MC files from data files.
        
        Creates separate entries with _mc suffix for MC files.
        
        Args:
            release_files: Dict mapping release years to file URLs
            
        Returns:
            Dict with MC files under "{year}_mc" keys and data under "{year}" keys
        """
        separated = {}
        
        for year, file_urls in release_files.items():
            mc_files = []
            data_files = []
            
            for url in file_urls:
                if "mc" in url.lower():
                    mc_files.append(url)
                else:
                    data_files.append(url)
            
            # Add data files (if any)
            if data_files:
                separated[year] = data_files
            
            # Add MC files with _mc suffix (if any)
            if mc_files:
                separated[f"{year}_mc"] = mc_files
        
        return separated
    
    def fetch_mc_metadata(self, dataset_id) -> Optional[MCDatasetMetadata]:
        """
        Fetch Monte-Carlo metadata for a single dataset (DSID).

        Uses atlasopenmagic ``get_metadata`` which returns all metadata fields
        for the dataset. Fields missing from the source (e.g. kFactor,
        genFiltEff on some samples) fall back to physics-safe defaults (1.0).

        Args:
            dataset_id: The dataset number (DSID), as int or str.

        Returns:
            MCDatasetMetadata for the dataset, or None if metadata could not be
            fetched or lacks the fields required to normalize the sample.
        """
        try:
            raw = atom.get_metadata(str(dataset_id))
        except Exception as e:
            logging.warning(f"Could not fetch MC metadata for dataset {dataset_id}: {e}")
            return None

        if not raw:
            logging.warning(f"No MC metadata returned for dataset {dataset_id}")
            return None

        cross_section_pb = self._to_float(raw.get("cross_section_pb"))
        sum_of_weights = self._to_float(raw.get("sumOfWeights"))

        if cross_section_pb is None or sum_of_weights is None or sum_of_weights == 0:
            logging.warning(
                f"Dataset {dataset_id} missing required normalization fields "
                f"(cross_section_pb={cross_section_pb}, sumOfWeights={sum_of_weights}); skipping"
            )
            return None

        # k-factor and filter efficiency default to 1.0 when absent (no correction).
        k_factor = self._to_float(raw.get("kFactor"))
        gen_filt_eff = self._to_float(raw.get("genFiltEff"))
        n_events = self._to_int(raw.get("nEvents"))
        dataset_number = self._to_int(raw.get("dataset_number")) or int(dataset_id)

        return MCDatasetMetadata(
            dataset_number=dataset_number,
            cross_section_pb=cross_section_pb,
            sum_of_weights=sum_of_weights,
            k_factor=k_factor if k_factor is not None else 1.0,
            gen_filt_eff=gen_filt_eff if gen_filt_eff is not None else 1.0,
            n_events=n_events,
            physics_short=raw.get("physics_short"),
            generator=raw.get("generator"),
        )

    def fetch_mc_metadata_for_datasets(
        self,
        dataset_ids,
        require_metadata: bool = False,
    ) -> dict:
        """
        Fetch MC metadata for many datasets (DSIDs).

        Args:
            dataset_ids: Iterable of dataset numbers (DSIDs).
            require_metadata: When True, raise if any dataset lacks the required
                metadata to normalize it (cross_section_pb, sumOfWeights).
                When False, such datasets are skipped and simply omitted.

        Returns:
            Dict mapping dataset_number (int) -> MCDatasetMetadata for every
            dataset that resolved successfully.

        Raises:
            ValueError: If require_metadata is True and one or more datasets
                could not be resolved.
        """
        resolved = {}
        missing = []

        for dataset_id in dataset_ids:
            md = self.fetch_mc_metadata(dataset_id)
            if md is None:
                missing.append(dataset_id)
            else:
                resolved[md.dataset_number] = md

        if missing and require_metadata:
            raise ValueError(
                f"require_metadata is enabled but {len(missing)} dataset(s) lack "
                f"required normalization metadata: {missing}"
            )
        if missing:
            logging.warning(
                f"Skipping {len(missing)} dataset(s) without required metadata: {missing}"
            )

        return resolved

    @staticmethod
    def _to_float(value) -> Optional[float]:
        """Coerce a metadata value to float, returning None if not possible."""
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value) -> Optional[int]:
        """Coerce a metadata value to int, returning None if not possible."""
        if value is None or value == "":
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

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
