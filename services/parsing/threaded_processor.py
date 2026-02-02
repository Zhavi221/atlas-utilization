"""
ThreadedFileProcessor - Orchestrates concurrent file parsing.

Single responsibility: Manage thread pool for parsing multiple files concurrently.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Optional, Callable
from tqdm import tqdm

from domain.events import EventBatch
from .file_parser import FileParser


class ThreadedFileProcessor:
    """
    Service for processing multiple files concurrently using thread pool.
    
    Coordinates FileParser instances across threads and yields results
    as they complete.
    """
    
    def __init__(
        self,
        file_parser: FileParser,
        max_threads: int,
        show_progress: bool = True
    ):
        """
        Initialize threaded processor.
        
        Args:
            file_parser: FileParser instance to use
            max_threads: Maximum number of concurrent threads
            show_progress: Whether to show progress bar
        """
        if max_threads <= 0:
            raise ValueError(f"max_threads must be positive, got {max_threads}")
        
        self.file_parser = file_parser
        self.max_threads = max_threads
        self.show_progress = show_progress
    
    def process_files(
        self,
        file_urls: list[str],
        tree_names: list[str],
        release_year: str,
        batch_size: int = 40_000,
        on_success: Optional[Callable[[str, int, float], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None
    ) -> Iterator[EventBatch]:
        """
        Process multiple files concurrently and yield EventBatch objects.
        
        Args:
            file_urls: List of file URLs to process
            tree_names: List of possible tree names
            release_year: Release year for schema lookup
            batch_size: Batch size for reading large files
            on_success: Optional callback(file_url, event_count, time_sec) on success
            on_error: Optional callback(file_url, exception) on error
            
        Yields:
            EventBatch objects as files are successfully parsed
        """
        total_files = len(file_urls)
        
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # Submit all parse jobs
            futures = {
                executor.submit(
                    self._parse_single_file,
                    file_url,
                    tree_names,
                    release_year,
                    batch_size
                ): file_url
                for file_url in file_urls
            }
            
            # Process results as they complete
            progress_bar = self._create_progress_bar(total_files)
            
            with progress_bar as pbar:
                for future in as_completed(futures):
                    file_url = futures[future]
                    
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per file
                        
                        if result is not None:
                            events, processing_time = result
                            
                            # Create EventBatch
                            batch = self._create_event_batch(
                                events=events,
                                file_url=file_url,
                                release_year=release_year,
                                processing_time=processing_time
                            )
                            
                            # Callback
                            if on_success:
                                on_success(file_url, batch.event_count, processing_time)
                            
                            yield batch
                    
                    except Exception as e:
                        logging.warning(f"Error processing file {file_url}: {e}")
                        if on_error:
                            on_error(file_url, e)
                    
                    finally:
                        if self.show_progress:
                            pbar.update(1)
    
    def _parse_single_file(
        self,
        file_url: str,
        tree_names: list[str],
        release_year: str,
        batch_size: int
    ) -> Optional[tuple]:
        """
        Parse a single file (runs in thread).
        
        Args:
            file_url: File URL to parse
            tree_names: List of possible tree names
            release_year: Release year
            batch_size: Batch size for reading
            
        Returns:
            Tuple of (events, processing_time) or None if parsing failed
        """
        import time
        start_time = time.time()
        
        events = self.file_parser.parse_file(
            file_path=file_url,
            tree_names=tree_names,
            release_year=release_year,
            batch_size=batch_size
        )
        
        processing_time = time.time() - start_time
        
        if events is None:
            return None
        
        return (events, processing_time)
    
    def _create_event_batch(
        self,
        events,
        file_url: str,
        release_year: str,
        processing_time: float
    ) -> EventBatch:
        """
        Create EventBatch from parsed events.
        
        Args:
            events: Awkward array of events
            file_url: Source file URL
            release_year: Release year
            processing_time: Time taken to parse
            
        Returns:
            EventBatch object
        """
        # Extract file ID from URL (use hash for now)
        file_id = hash(file_url)
        
        # Calculate size
        size_bytes = events.layout.nbytes if hasattr(events, 'layout') else 0
        event_count = len(events) if hasattr(events, '__len__') else 0
        
        return EventBatch(
            events=events,
            file_id=file_id,
            release_year=release_year,
            size_bytes=size_bytes,
            event_count=event_count,
            processing_time_sec=processing_time
        )
    
    def _create_progress_bar(self, total: int):
        """
        Create progress bar or no-op context manager.
        
        Args:
            total: Total number of items
            
        Returns:
            Progress bar context manager or no-op
        """
        if self.show_progress:
            return tqdm(
                total=total,
                desc="Processing files",
                unit="file",
                dynamic_ncols=True,
                mininterval=1
            )
        else:
            # No-op context manager
            from contextlib import nullcontext
            return nullcontext()


class ParsingStatisticsCollector:
    """
    Helper class to collect statistics during parsing.
    
    Thread-safe collector that aggregates parsing statistics.
    """
    
    def __init__(self):
        """Initialize statistics collector."""
        import threading
        self.lock = threading.Lock()
        self.successful_count = 0
        self.failed_count = 0
        self.total_events = 0
        self.total_size_bytes = 0
        self.failed_files = []
        self.processing_times = []
    
    def record_success(self, file_url: str, event_count: int, size_bytes: int, time_sec: float):
        """Record a successful parse."""
        with self.lock:
            self.successful_count += 1
            self.total_events += event_count
            self.total_size_bytes += size_bytes
            self.processing_times.append(time_sec)
    
    def record_failure(self, file_url: str, error: Exception):
        """Record a failed parse."""
        with self.lock:
            self.failed_count += 1
            self.failed_files.append((file_url, str(error)))
    
    def get_summary(self) -> dict:
        """Get statistics summary."""
        with self.lock:
            total = self.successful_count + self.failed_count
            avg_time = (
                sum(self.processing_times) / len(self.processing_times)
                if self.processing_times else 0
            )
            
            return {
                "total_files": total,
                "successful_files": self.successful_count,
                "failed_files": self.failed_count,
                "success_rate": (
                    (self.successful_count / total * 100) if total > 0 else 0
                ),
                "total_events": self.total_events,
                "total_size_mb": self.total_size_bytes / (1024 * 1024),
                "average_processing_time_sec": avg_time,
                "failed_file_list": self.failed_files
            }
