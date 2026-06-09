"""
FetchMetadataHandler - Handles metadata fetching state.
Fetches file URLs from ATLAS Open Data API.
"""
from orchestration.context import PipelineContext
from orchestration.states import PipelineState
from .base import StateHandler
from services.metadata.fetcher import MetadataFetcher, _classify_url, UrlType
from services.metadata.cache import MetadataCache


class FetchMetadataHandler(StateHandler):

    def __init__(
        self,
        metadata_fetcher: MetadataFetcher,
        metadata_cache: MetadataCache
    ):
        super().__init__()
        self.fetcher = metadata_fetcher
        self.cache = metadata_cache

    def handle(self, context: PipelineContext) -> tuple[PipelineContext, PipelineState]:
        self._log_state_entry(context)

        parsing_config = context.config.parsing_config
        if not parsing_config:
            next_state = self._determine_next_state(context)
            self._log_state_exit(context, next_state)
            return context, next_state

        metadata = self.cache.load()

        if metadata:
            self.logger.info(f"Loaded metadata from cache: {self.cache.cache_path}")

            # ── CHANGE 6: Validate cache integrity before using it ────────────
            # BEFORE: cache was used immediately after loading with no checks.
            # A cache built with parse_mc=False contains MC urls in data keys,
            # which causes MC events to be parsed as collision data silently.
            #
            # AFTER: validate and abort with a clear error if contaminated.
            self._validate_cache_or_abort(metadata)
            # ── END CHANGE 6 ─────────────────────────────────────────────────

        else:
            self.logger.info("Cache miss, fetching metadata from API...")

            # ── CHANGE 7: Remove `separate_mc` argument from fetcher call ─────
            # BEFORE:
            #     metadata = self.fetcher.fetch(
            #         release_years=...,
            #         record_ids=...,
            #         separate_mc=parsing_config.parse_mc   # <-- removed
            #     )
            # AFTER: separation always happens inside fetcher.fetch()
            metadata = self.fetcher.fetch(
                release_years=list(parsing_config.release_years) if parsing_config.release_years else None,
                record_ids=list(parsing_config.specific_record_ids) if parsing_config.specific_record_ids else None,
            )
            # ── END CHANGE 7 ─────────────────────────────────────────────────

            try:
                self.cache.save(metadata)
            except TimeoutError as e:
                self.logger.warning(f"Could not save to cache: {e}")

        total_files = sum(len(urls) for urls in metadata.values())
        self.logger.info(
            f"Fetched metadata: {len(metadata)} release year(s), {total_files} total files"
        )

        updated_context = context.with_metadata(metadata)
        next_state = self._determine_next_state(updated_context)
        self._log_state_exit(context, next_state)
        return updated_context, next_state

    # ── CHANGE 8: Add cache validation method ────────────────────────────────
    def _validate_cache_or_abort(self, metadata: dict) -> None:
        """
        Validate that no MC URLs are in data keys and vice versa.

        Raises RuntimeError with a clear message if contamination is found,
        rather than silently proceeding with bad data.

        To fix a contaminated cache: delete the cache file and re-run.
        The fetcher will rebuild it correctly with the patched _separate_mc_files.
        """
        contaminated_keys = []

        for key, urls in metadata.items():
            is_mc_key = key.endswith("_mc")
            for url in urls:
                try:
                    url_type = _classify_url(url)
                except ValueError:
                    contaminated_keys.append(
                        f"  Key '{key}': unclassifiable URL: {url}"
                    )
                    continue

                if is_mc_key and url_type != UrlType.MC:
                    contaminated_keys.append(
                        f"  Key '{key}' (MC key): contains DATA url: {url}"
                    )
                elif not is_mc_key and url_type != UrlType.DATA:
                    contaminated_keys.append(
                        f"  Key '{key}' (data key): contains MC url: {url}"
                    )

                # Stop scanning after first 5 violations — enough to diagnose
                if len(contaminated_keys) >= 5:
                    break
            if len(contaminated_keys) >= 5:
                break

        if contaminated_keys:
            raise RuntimeError(
                f"Contaminated metadata cache detected at: {self.cache.cache_path}\n"
                f"MC and data URLs are mixed. This cache was built with parse_mc=False.\n"
                f"Fix: delete the cache file and re-run to rebuild it.\n"
                f"First violations found:\n"
                + "\n".join(contaminated_keys)
            )

        self.logger.info("Cache integrity check passed — no cross-contamination.")
    # ── END CHANGE 8 ─────────────────────────────────────────────────────────
