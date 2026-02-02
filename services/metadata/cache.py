"""
MetadataCache service - Caches file metadata to disk.

Single responsibility: Load/save metadata cache with file locking.
"""

import json
import os
import time
import logging
from typing import Optional


class MetadataCache:
    """
    Service for caching metadata to disk with file locking.
    
    Provides atomic file operations with exclusive locking to prevent
    race conditions in multi-process environments.
    """
    
    def __init__(self, cache_path: str, max_wait_time: int = 300):
        """
        Initialize metadata cache.
        
        Args:
            cache_path: Path to cache file
            max_wait_time: Maximum seconds to wait for lock (default 5 minutes)
        """
        self.cache_path = cache_path
        self.lock_path = f"{cache_path}.lock"
        self.max_wait_time = max_wait_time
        self.wait_interval = 2  # Check every 2 seconds
    
    def load(self) -> Optional[dict]:
        """
        Load cached metadata from disk.
        
        Returns:
            Dict of cached metadata, or None if cache doesn't exist or is invalid
        """
        if not os.path.exists(self.cache_path):
            return None
        
        try:
            with open(self.cache_path, "r") as f:
                data = json.load(f)
            logging.info(f"Loaded metadata from cache: {self.cache_path}")
            return data
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to load cache from {self.cache_path}: {e}")
            return None
    
    def save(self, metadata: dict) -> bool:
        """
        Save metadata to disk with atomic file operations and locking.
        
        Uses exclusive file locking to prevent race conditions.
        If lock cannot be acquired within max_wait_time, raises TimeoutError.
        
        Args:
            metadata: Dict of metadata to save
            
        Returns:
            True if save succeeded, False otherwise
            
        Raises:
            TimeoutError: If lock cannot be acquired within max_wait_time
        """
        # Try to acquire lock
        lock_acquired = self._acquire_lock()
        
        if not lock_acquired:
            raise TimeoutError(
                f"Could not acquire lock for {self.cache_path} "
                f"after {self.max_wait_time} seconds"
            )
        
        try:
            # Write atomically using temp file + rename
            temp_path = f"{self.cache_path}.tmp"
            
            with open(temp_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Atomic rename
            os.rename(temp_path, self.cache_path)
            
            logging.info(f"Saved metadata to cache: {self.cache_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save cache to {self.cache_path}: {e}")
            
            # Clean up temp file if it exists
            temp_path = f"{self.cache_path}.tmp"
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            return False
            
        finally:
            self._release_lock()
    
    def _acquire_lock(self) -> bool:
        """
        Try to acquire exclusive lock on cache file.
        
        Waits up to max_wait_time for lock to become available.
        
        Returns:
            True if lock acquired, False if timeout
        """
        elapsed = 0
        
        while elapsed < self.max_wait_time:
            try:
                # Try to create lock file exclusively (atomic operation)
                lock_fd = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
                # Got the lock!
                os.close(lock_fd)
                return True
                
            except FileExistsError:
                # Lock file exists - another process has the lock
                logging.debug(
                    f"Lock file exists, waiting... (waited {elapsed}s)"
                )
                time.sleep(self.wait_interval)
                elapsed += self.wait_interval
                
                # Check if cache was created while we waited
                if os.path.exists(self.cache_path):
                    # Another process created the cache, we can stop waiting
                    logging.info("Cache was created by another process")
                    return False
                    
            except Exception as e:
                logging.error(f"Unexpected error acquiring lock: {e}")
                return False
        
        # Timeout
        return False
    
    def _release_lock(self):
        """Release the lock by removing lock file."""
        if os.path.exists(self.lock_path):
            try:
                os.unlink(self.lock_path)
            except Exception as e:
                logging.warning(f"Failed to release lock: {e}")
    
    def clear(self) -> bool:
        """
        Clear the cache by removing cache file.
        
        Returns:
            True if cache was cleared, False otherwise
        """
        if not os.path.exists(self.cache_path):
            return True
        
        try:
            os.remove(self.cache_path)
            logging.info(f"Cleared cache: {self.cache_path}")
            return True
        except Exception as e:
            logging.warning(f"Failed to clear cache {self.cache_path}: {e}")
            return False
    
    def exists(self) -> bool:
        """Check if cache file exists."""
        return os.path.exists(self.cache_path)
    
    def wait_for_cache(self) -> Optional[dict]:
        """
        Wait for cache to be created by another process.
        
        Useful in multi-process scenarios where one process creates
        the cache and others wait for it.
        
        Returns:
            Cached metadata dict, or None if timeout
        """
        elapsed = 0
        
        while elapsed < self.max_wait_time:
            if self.exists():
                return self.load()
            
            logging.debug(f"Waiting for cache to be created... (waited {elapsed}s)")
            time.sleep(self.wait_interval)
            elapsed += self.wait_interval
        
        logging.warning(f"Timeout waiting for cache at {self.cache_path}")
        return None
